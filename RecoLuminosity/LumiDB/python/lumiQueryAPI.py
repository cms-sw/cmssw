import os
import coral,datetime
from RecoLuminosity.LumiDB import nameDealer,lumiTime,CommonUtil
import array
from RecoLuminosity.LumiDB import argparse, nameDealer, selectionParser, hltTrgSeedMapper, \
     connectstrParser, cacheconfigParser, tablePrinter, csvReporter, csvSelectionParser
from RecoLuminosity.LumiDB.wordWrappers import wrap_always, wrap_onspace, wrap_onspace_strict
from pprint import pprint, pformat

'''
This module defines lowlevel SQL query API for lumiDB 
We do not like range queries so far because of performance of range scan.Use only necessary.
The principle is to query by runnumber and per each coral queryhandle
Try reuse db session/transaction and just renew query handle each time to reduce metadata queries.
Avoid unnecessary explicit order by
Do not handle transaction in here.
Do not do explicit del queryhandle in here.
Note: all the returned dict format are not sorted by itself.Sort it outside if needed.
'''
###==============temporarilly here======###

class ParametersObject (object):

    def __init__ (self):
        self.norm            = 1.0
        self.lumiversion     = '0001'
        self.NBX             = 3564  # number beam crossings
        self.rotationRate    = 11245.613 # for 3.5 TeV Beam energy
        self.normFactor      = 6.37
        self.beammode        = '' #possible choices stable, quiet, either
        self.verbose         = False
        self.noWarnings      = False
        self.lumischema      = 'CMS_LUMI_PROD'
        #self.lumidb          = 'oracle://cms_orcoff_prod/cms_lumi_prod'
        self.lumisummaryname = 'LUMISUMMARY'
        self.lumidetailname  = 'LUMIDETAIL'
        self.lumiXing        = False
        self.xingMinLum      = 1.0e-4
        self.xingIndex       = 5
        self.minBiasXsec     = 71300 # unit: microbarn
        self.pileupHistName  = 'pileup'
        self.maxPileupBin    = 10
        self.calculateTimeParameters()

    def calculateTimeParameters (self):
        '''Given the rotation rate, calculate lumi section length and
        rotation time.  This should be called if rotationRate is
        updated.'''
        self.rotationTime    = 1 / self.rotationRate
        self.lumiSectionLen  = 2**18 * self.rotationTime
        
    def defaultfrontierConfigString (self):
        return '''<frontier-connect><proxy url = "http://cmst0frontier.cern.ch:3128"/><proxy url = "http://cmst0frontier.cern.ch:3128"/><proxy url = "http://cmst0frontier1.cern.ch:3128"/><proxy url = "http://cmst0frontier2.cern.ch:3128"/><server url = "http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url = "http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url = "http://cmsfrontier1.cern.ch:8000/FrontierInt"/><server url = "http://cmsfrontier2.cern.ch:8000/FrontierInt"/><server url = "http://cmsfrontier3.cern.ch:8000/FrontierInt"/><server url = "http://cmsfrontier4.cern.ch:8000/FrontierInt"/></frontier-connect>'''

def lslengthsec (numorbit, numbx):
    #print numorbit, numbx
    l = numorbit * numbx * 25.0e-09
    return l

def lsBylsLumi (deadtable):
    """
    input: {lsnum:[deadtime, instlumi, bit_0, norbits,prescale...]}
    output: {lsnum:[instlumi, recordedlumi...]}
    """
    result = {}
    for myls, deadArray in deadtable.items():
        lstime = lslengthsec (deadArray[3], 3564)
        instlumi = deadArray[1] * lstime
        if float( deadArray[2] ) ==  0.0:
            deadfrac = 1.0
        else:
            deadfrac = float (deadArray[0]) / (float (deadArray[2])*float(deadArray[4]))
        recordedLumi = instlumi * (1.0 - deadfrac)
        myLsList = [instlumi, recordedLumi]
        #print myls,instlumi,recordedLumi,lstime,deadfrac
        if len (deadArray) > 5:
            myLsList.extend (deadArray[5:])
        result[myls] = myLsList
    return result


def deliveredLumiForRange (dbsession, parameters, inputRange):
    '''Takes either single run as a string or dictionary of run ranges'''
    lumidata = []
    # is this a single string?
    if isinstance (inputRange, str):
        lumidata.append( deliveredLumiForRun (dbsession, parameters, inputRange) )
    else:
        # if not, it's one of these dictionary things
        for run in sorted( inputRange.runs() ):
            if parameters.verbose:
                print "run", run
            lumidata.append( deliveredLumiForRun (dbsession, parameters, run) )
    #print lumidata
    return lumidata


def recordedLumiForRange (dbsession, parameters, inputRange):
    '''Takes either single run as a string or dictionary of run ranges'''
    lumidata = []
    # is this a single string?
    if isinstance (inputRange, str):
        lumiDataPiece = recordedLumiForRun (dbsession, parameters, inputRange)
        if parameters.lumiXing:
            # get the xing information for the run
            xingLumiDict = xingLuminosityForRun (dbsession, inputRange,
                                                 parameters)
            mergeXingLumi (lumiDataPiece, xingLumiDict)
        lumidata.append (lumiDataPiece)
        
    else:
        # we want to collapse the lists so that every run is considered once.
        runLsDict = {}
        maxLumiSectionDict = {}
        for (run, lslist) in sorted (inputRange.runsandls().items() ):
            if(len(lslist)!=0):
                maxLumiSectionDict[run] = max ( max (lslist),
                                           maxLumiSectionDict.get(run,0) )
            runLsDict.setdefault (run, []).append (lslist)
        for run, metaLsList in sorted (runLsDict.iteritems()):
            if parameters.verbose:
                print "run", run
            runLumiData = []
            for lslist in metaLsList:
                runLumiData.append( recordedLumiForRun (dbsession, parameters,
                                                        run, lslist) )
            if parameters.lumiXing:
                # get the xing information once for the whole run
                xingLumiDict = xingLuminosityForRun (dbsession, run,
                                                     parameters,
                                                     maxLumiSection = \
                                                     maxLumiSectionDict[run])
                # merge it with every piece of lumi data for this run
                for lumiDataPiece in runLumiData:
                    mergeXingLumi (lumiDataPiece, xingLumiDict)
                    lumidata.append (lumiDataPiece)
            else:
                lumidata.extend( runLumiData )
    return lumidata



def deliveredLumiForRun (dbsession, parameters, runnum):    
    """
    select sum (INSTLUMI), count (INSTLUMI) from lumisummary where runnum = 124025 and lumiversion = '0001';
    select INSTLUMI,NUMORBIT  from lumisummary where runnum = 124025 and lumiversion = '0001'
    query result unit E27cm^-2 (= 1 / mb)"""    
    #if parameters.verbose:
    #    print 'deliveredLumiForRun : norm : ', parameters.norm, ' : run : ', runnum
    #output ['run', 'totalls', 'delivered', 'beammode']
    delivered = 0.0
    totalls = 0
    try:
        conditionstring="RUNNUM = :runnum AND LUMIVERSION = :lumiversion"
        dbsession.transaction().start (True)
        schema = dbsession.nominalSchema()
        query = schema.tableHandle (nameDealer.lumisummaryTableName()).newQuery()
        #query.addToOutputList ("sum (INSTLUMI)", "totallumi")
        #query.addToOutputList ("count (INSTLUMI)", "totalls")
        query.addToOutputList("INSTLUMI",'instlumi')
        query.addToOutputList ("NUMORBIT", "norbits")
        queryBind = coral.AttributeList()
        queryBind.extend ("runnum", "unsigned int")
        queryBind.extend ("lumiversion", "string")
        queryBind["runnum"].setData (int (runnum))
        queryBind["lumiversion"].setData (parameters.lumiversion)
        #print parameters.beammode
        if len(parameters.beammode)!=0:
            conditionstring=conditionstring+' and BEAMSTATUS=:beamstatus'
            queryBind.extend('beamstatus','string')
            queryBind['beamstatus'].setData(parameters.beammode)
        result = coral.AttributeList()
        result.extend ("instlumi", "float")
        result.extend ("norbits", "unsigned int")
        query.defineOutput (result)
        query.setCondition (conditionstring,queryBind)
        #query.limitReturnedRows (1)
        #query.groupBy ('NUMORBIT')
        cursor = query.execute()
        while cursor.next():
            instlumi = cursor.currentRow()['instlumi'].data()
            norbits = cursor.currentRow()['norbits'].data()

            if instlumi is not None and norbits is not None:
                lstime = lslengthsec(norbits, parameters.NBX)
                delivered=delivered+instlumi*parameters.norm*lstime
                totalls+=1
        del query
        dbsession.transaction().commit()
        lumidata = []
        if delivered == 0.0:
            lumidata = [str (runnum), 'N/A', 'N/A', 'N/A']
        else:
            lumidata = [str (runnum), str (totalls), '%.3f'%delivered, parameters.beammode]
        return lumidata
    except Exception, e:
        print str (e)
        dbsession.transaction().rollback()
        del dbsession

def recordedLumiForRun (dbsession, parameters, runnum, lslist = None):
    """
    lslist = [] means take none in the db
    lslist = None means to take all in the db
    output: ['runnumber', 'trgtable{}', 'deadtable{}']
    """
    recorded = 0.0
    lumidata = [] #[runnumber, trgtable, deadtable]
    trgtable = {} #{hltpath:[l1seed, hltprescale, l1prescale]}
    deadtable = {} #{lsnum:[deadtime, instlumi, bit_0, norbits,bitzero_prescale]}
    lumidata.append (runnum)
    lumidata.append (trgtable)
    lumidata.append (deadtable)
    collectedseeds = [] #[ (hltpath, l1seed)]
    conditionstring='trghltmap.HLTKEY = cmsrunsummary.HLTKEY AND cmsrunsummary.RUNNUM = :runnumber'
    try:
        dbsession.transaction().start (True)
        schema = dbsession.nominalSchema()
        query = schema.newQuery()
        query.addToTableList (nameDealer.cmsrunsummaryTableName(), 'cmsrunsummary')
        query.addToTableList (nameDealer.trghltMapTableName(), 'trghltmap')#small table first
        queryCondition = coral.AttributeList()
        queryCondition.extend ("runnumber", "unsigned int")
        queryCondition["runnumber"].setData (int (runnum))
        query.setCondition (conditionstring,queryCondition)
        query.addToOutputList ("trghltmap.HLTPATHNAME", "hltpathname")
        query.addToOutputList ("trghltmap.L1SEED", "l1seed")
        result = coral.AttributeList()
        result.extend ("hltpathname", "string")
        result.extend ("l1seed", "string")
        query.defineOutput (result)
        cursor = query.execute()
        while cursor.next():
            hltpathname = cursor.currentRow()["hltpathname"].data()
            l1seed = cursor.currentRow()["l1seed"].data()
            collectedseeds.append ( (hltpathname, l1seed))
        #print 'collectedseeds ', collectedseeds
        del query
        dbsession.transaction().commit()
        #loop over hltpath
        for (hname, sname) in collectedseeds:
            l1bitname = hltTrgSeedMapper.findUniqueSeed (hname, sname)
            #print 'found unque seed ', hname, l1bitname
            if l1bitname:
                lumidata[1][hname] = []
                lumidata[1][hname].append (l1bitname.replace ('\"', ''))
        dbsession.transaction().start (True)
        schema = dbsession.nominalSchema()
        hltprescQuery = schema.tableHandle (nameDealer.hltTableName()).newQuery()
        hltprescQuery.addToOutputList ("PATHNAME", "hltpath")
        hltprescQuery.addToOutputList ("PRESCALE", "hltprescale")
        hltprescCondition = coral.AttributeList()
        hltprescCondition.extend ('runnumber', 'unsigned int')
        hltprescCondition.extend ('cmslsnum', 'unsigned int')
        hltprescCondition.extend ('inf', 'unsigned int')
        hltprescResult = coral.AttributeList()
        hltprescResult.extend ('hltpath', 'string')
        hltprescResult.extend ('hltprescale', 'unsigned int')
        hltprescQuery.defineOutput (hltprescResult)
        hltprescCondition['runnumber'].setData (int (runnum))
        hltprescCondition['cmslsnum'].setData (1)
        hltprescCondition['inf'].setData (0)
        hltprescQuery.setCondition ("RUNNUM = :runnumber and CMSLSNUM = :cmslsnum and PRESCALE != :inf",
                                    hltprescCondition)
        cursor = hltprescQuery.execute()
        while cursor.next():
            hltpath = cursor.currentRow()['hltpath'].data()
            hltprescale = cursor.currentRow()['hltprescale'].data()
            if lumidata[1].has_key (hltpath):
                lumidata[1][hltpath].append (hltprescale)
                
        cursor.close()
        del hltprescQuery
        dbsession.transaction().commit()      
        dbsession.transaction().start (True)
        schema = dbsession.nominalSchema()
        query = schema.newQuery()
        query.addToTableList (nameDealer.trgTableName(), 'trg')
        query.addToTableList (nameDealer.lumisummaryTableName(), 'lumisummary')#small table first--right-most
        queryCondition = coral.AttributeList()
        queryCondition.extend ("runnumber", "unsigned int")
        queryCondition.extend ("lumiversion", "string")
        queryCondition["runnumber"].setData (int (runnum))
        queryCondition["lumiversion"].setData (parameters.lumiversion)
        conditionstring='lumisummary.RUNNUM =:runnumber and lumisummary.LUMIVERSION =:lumiversion AND lumisummary.CMSLSNUM=trg.CMSLSNUM and lumisummary.RUNNUM=trg.RUNNUM'
        if len(parameters.beammode)!=0:
            conditionstring=conditionstring+' and lumisummary.BEAMSTATUS=:beamstatus'
            queryCondition.extend('beamstatus','string')
            queryCondition['beamstatus'].setData(parameters.beammode)
        query.setCondition(conditionstring,queryCondition)
        query.addToOutputList ("lumisummary.CMSLSNUM", "cmsls")
        query.addToOutputList ("lumisummary.INSTLUMI", "instlumi")
        query.addToOutputList ("lumisummary.NUMORBIT", "norbits")
        query.addToOutputList ("trg.TRGCOUNT",         "trgcount")
        query.addToOutputList ("trg.BITNAME",          "bitname")
        query.addToOutputList ("trg.DEADTIME",         "trgdeadtime")
        query.addToOutputList ("trg.PRESCALE",         "trgprescale")
        query.addToOutputList ("trg.BITNUM",           "trgbitnum")
        
        result = coral.AttributeList()
        result.extend ("cmsls",       "unsigned int")
        result.extend ("instlumi",    "float")
        result.extend ("norbits",     "unsigned int")
        result.extend ("trgcount",    "unsigned int")
        result.extend ("bitname",     "string")
        result.extend ("trgdeadtime", "unsigned long long")
        result.extend ("trgprescale", "unsigned int")
        result.extend ("trgbitnum",   "unsigned int")
        trgprescalemap = {}
        query.defineOutput (result)
        cursor = query.execute()
        while cursor.next():
            cmsls       = cursor.currentRow()["cmsls"].data()
            instlumi    = cursor.currentRow()["instlumi"].data()*parameters.norm
            norbits     = cursor.currentRow()["norbits"].data()
            trgcount    = cursor.currentRow()["trgcount"].data()
            trgbitname  = cursor.currentRow()["bitname"].data()
            trgdeadtime = cursor.currentRow()["trgdeadtime"].data()
            trgprescale = cursor.currentRow()["trgprescale"].data()
            trgbitnum   = cursor.currentRow()["trgbitnum"].data()
            if cmsls == 1:
                if not trgprescalemap.has_key (trgbitname):
                    trgprescalemap[trgbitname] = trgprescale
            if trgbitnum == 0:
                if not deadtable.has_key (cmsls):
                    deadtable[cmsls] = []
                    deadtable[cmsls].append (trgdeadtime)
                    deadtable[cmsls].append (instlumi)
                    deadtable[cmsls].append (trgcount)
                    deadtable[cmsls].append (norbits)
                    deadtable[cmsls].append (trgprescale)
        cursor.close()
        del query
        dbsession.transaction().commit()
        
        #
        #consolidate results
        #
        #trgtable
        #print 'trgprescalemap', trgprescalemap
        #print lumidata[1]
        for hpath, trgdataseq in lumidata[1].items():   
            bitn = trgdataseq[0]
            if trgprescalemap.has_key (bitn) and len (trgdataseq) == 2:
                lumidata[1][hpath].append (trgprescalemap[bitn])                
        #filter selected cmsls
        lumidata[2] = filterDeadtable (deadtable, lslist)
        #print 'lslist ',lslist
        if not parameters.noWarnings:
            if len(lumidata[2])!=0:
                for lumi, deaddata in lumidata[2].items():
                    if deaddata[1] == 0.0 and deaddata[2]!=0 and deaddata[0]!=0:
                        print '[Warning] : run %s :ls %d has 0 instlumi but trigger has data' % (runnum, lumi)
                    if (deaddata[2] == 0 or deaddata[0] == 0) and deaddata[1]!=0.0:
                        print '[Warning] : run %s :ls %d has 0 dead counts or 0 zerobias bit counts, but inst!=0' % (runnum, lumi)
        #print 'lumidata[2] ', lumidata[2]
    except Exception, e:
        print str (e)
        dbsession.transaction().rollback()
        del dbsession
    #print 'before return lumidata ', lumidata
    ## if parameters.lumiXing:
    ##     xingLumiDict =  xingLuminosityForRun (dbsession, runnum, parameters)
    ##     mergeXingLumi (lumidata, xingLumiDict)
    return lumidata


def filterDeadtable (inTable, lslist):
    result = {}
    if lslist is None:
        return inTable
    if len (lslist) == 0: #if request no ls, then return nothing
        return result
    for existingLS in inTable.keys():
        if existingLS in lslist:
            result[existingLS] = inTable[existingLS]
    return result


def printDeliveredLumi (lumidata, mode):
    labels = [ ('Run', 'Delivered LS', 'Delivered'+u' (/\u03bcb)'.encode ('utf-8'), 'Beam Mode')]
    print tablePrinter.indent (labels+lumidata, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'right',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace (x, 20) )

def dumpData (lumidata, filename):
    """
    input params: lumidata [{'fieldname':value}]
                  filename csvname
    """
    
    r = csvReporter.csvReporter(filename)
    r.writeRows(lumidata)

def calculateTotalRecorded (deadtable):
    """
    input: {lsnum:[deadtime, instlumi, bit_0, norbits,prescale]}
    output: recordedLumi
    """
    recordedLumi = 0.0
    for myls, d in deadtable.items():
        instLumi = d[1]
        #deadfrac = float (d[0])/float (d[2]*3564)
        #print myls, float (d[2])
        if float (d[2]) == 0.0:
            deadfrac = 1.0
        else:
            deadfrac = float (d[0])/(float (d[2])*float (d[-1]))
        lstime = lslengthsec (d[3], 3564)
        recordedLumi += instLumi* (1.0-deadfrac)*lstime
    return recordedLumi


def splitlistToRangeString (inPut):
    result = []
    first = inPut[0]
    last = inPut[0]
    result.append ([inPut[0]])
    counter = 0
    for i in inPut[1:]:
        if i == last+1:
            result[counter].append (i)
        else:
            counter += 1
            result.append ([i])
        last = i
    return ', '.join (['['+str (min (x))+'-'+str (max (x))+']' for x in result])


def calculateEffective (trgtable, totalrecorded):
    """
    input: trgtable{hltpath:[l1seed, hltprescale, l1prescale]}, totalrecorded (float)
    output:{hltpath, recorded}
    """
    #print 'inputtrgtable', trgtable
    result = {}
    for hltpath, data in trgtable.items():
        if len (data) ==  3:
            result[hltpath] = totalrecorded/ (data[1]*data[2])
        else:
            result[hltpath] = 0.0
    return result


def getDeadfractions (deadtable):
    """
    inputtable: {lsnum:[deadtime, instlumi, bit_0, norbits,bit_0_prescale]}
    output: {lsnum:deadfraction}
    """
    result = {}
    for myls, d in deadtable.items():
        #deadfrac = float (d[0])/ (float (d[2])*float (3564))
        if float (d[2]) == 0.0: ##no beam
            deadfrac = -1.0
        else:
            deadfrac = float (d[0])/ (float (d[2])*float(d[-1]))
        result[myls] = deadfrac
    return result

def printPerLSLumi (lumidata, isVerbose = False):
    '''
    input lumidata  [['runnumber', 'trgtable{}', 'deadtable{}']]
    deadtable {lsnum:[deadtime, instlumi, bit_0, norbits,prescale]}
    '''
    datatoprint = []
    totalrow = []
    labels = [ ('Run', 'LS', 'Delivered', 'Recorded'+u' (/\u03bcb)'.encode ('utf-8'))]
    lastrowlabels = [ ('Selected LS', 'Delivered'+u' (/\u03bcb)'.encode ('utf-8'), 'Recorded'+u' (/\u03bcb)'.encode ('utf-8'))]
    totalDeliveredLS = 0
    totalSelectedLS = 0
    totalDelivered = 0.0
    totalRecorded = 0.0
    
    for perrundata in lumidata:
        runnumber = perrundata[0]
        deadtable = perrundata[2]
        lumiresult = lsBylsLumi (deadtable)
        totalSelectedLS = totalSelectedLS+len (deadtable)
        for lsnum, dataperls in lumiresult.items():
            rowdata = []
            if len (dataperls) == 0:
                rowdata  +=  [str (runnumber), str (lsnum), 'N/A', 'N/A']
            else:
                rowdata  +=  [str (runnumber), str (lsnum), '%.3f' % (dataperls[0]), '%.3f' % (dataperls[1])]
                totalDelivered = totalDelivered+dataperls[0]
                totalRecorded = totalRecorded+dataperls[1]
            datatoprint.append (rowdata)
    totalrow.append ([str (totalSelectedLS), '%.3f'% (totalDelivered), '%.3f'% (totalRecorded)])
    print ' ==  = '
    print tablePrinter.indent (labels+datatoprint, hasHeader = True, separateRows = False, prefix = '| ',
                               postfix = ' |', justify = 'right', delim = ' | ',
                               wrapfunc = lambda x: wrap_onspace_strict (x, 22))
    print ' ==  =  Total : '
    print tablePrinter.indent (lastrowlabels+totalrow, hasHeader = True, separateRows = False, prefix = '| ',
                               postfix = ' |', justify = 'right', delim = ' | ',
                               wrapfunc = lambda x: wrap_onspace (x, 20))    

    
def dumpPerLSLumi (lumidata):
    datatodump = []
    for perrundata in lumidata:
        runnumber = perrundata[0]
        deadtable = perrundata[2]
        lumiresult = lsBylsLumi (deadtable)
        for lsnum, dataperls in lumiresult.items():
            rowdata = []
            if len (dataperls) == 0:
                rowdata += [str (runnumber), str (lsnum), 'N/A', 'N/A']
            else:
                rowdata += [str (runnumber), str (lsnum), dataperls[0], dataperls[1]]
            if len (dataperls) > 2:
                rowdata.extend ( flatten (dataperls[2:]) )
            datatodump.append (rowdata)
    return datatodump


def printRecordedLumi (lumidata, isVerbose = False, hltpath = ''):
    datatoprint = []
    totalrow = []
    labels = [ ('Run', 'HLT path', 'Recorded'+u' (/\u03bcb)'.encode ('utf-8'))]
    lastrowlabels = [ ('Selected LS', 'Recorded'+u' (/\u03bcb)'.encode ('utf-8'))]
    if len (hltpath) != 0 and hltpath != 'all':
        lastrowlabels = [ ('Selected LS', 'Recorded'+u' (/\u03bcb)'.encode ('utf-8'),
                           'Effective '+u' (/\u03bcb) '.encode ('utf-8')+hltpath)]
    if isVerbose:
        labels = [ ('Run', 'HLT-path', 'L1-bit', 'L1-presc', 'HLT-presc', 'Recorded'+u' (/\u03bcb)'.encode ('utf-8'))]
    totalSelectedLS = 0
    totalRecorded = 0.0
    totalRecordedInPath = 0.0
    
    for dataperRun in lumidata:
        runnum = dataperRun[0]
        if len (dataperRun[1]) == 0:
            rowdata = []
            rowdata += [str (runnum)]+2*['N/A']
            datatoprint.append (rowdata)
            continue
        perlsdata = dataperRun[2]
        totalSelectedLS = totalSelectedLS+len (perlsdata)
        recordedLumi = 0.0
        #norbits = perlsdata.values()[0][3]
        recordedLumi = calculateTotalRecorded (perlsdata)
        totalRecorded = totalRecorded+recordedLumi
        trgdict = dataperRun[1]
        effective = calculateEffective (trgdict, recordedLumi)
        if trgdict.has_key (hltpath) and effective.has_key (hltpath):
            rowdata = []
            l1bit = trgdict[hltpath][0]
            if len (trgdict[hltpath]) !=  3:
                if not isVerbose:
                    rowdata += [str (runnum), hltpath, 'N/A']
                else:
                    rowdata += [str (runnum), hltpath, l1bit, 'N/A', 'N/A', 'N/A']
            else:
                if not isVerbose:
                    rowdata += [str (runnum), hltpath, '%.3f'% (effective[hltpath])]
                else:
                    hltprescale = trgdict[hltpath][1]
                    l1prescale = trgdict[hltpath][2]
                    rowdata += [str (runnum), hltpath, l1bit, str (l1prescale), str (hltprescale),
                                '%.3f'% (effective[hltpath])]
                totalRecordedInPath = totalRecordedInPath+effective[hltpath]
            datatoprint.append (rowdata)
            continue
        
        for trg, trgdata in trgdict.items():
            #print trg, trgdata
            rowdata = []                    
            if trg == trgdict.keys()[0]:
                rowdata += [str (runnum)]
            else:
                rowdata += ['']
            l1bit = trgdata[0]
            if len (trgdata) == 3:
                if not isVerbose:
                    rowdata += [trg, '%.3f'% (effective[trg])]
                else:
                    hltprescale = trgdata[1]
                    l1prescale = trgdata[2]
                    rowdata += [trg, l1bit, str (l1prescale), str (hltprescale), '%.3f'% (effective[trg])]
            else:
                if not isVerbose:
                    rowdata += [trg, 'N/A']
                else:
                    rowdata += [trg, l1bit, 'N/A', 'N/A', '%.3f'% (effective[trg])]
            datatoprint.append (rowdata)
    #print datatoprint
    print ' ==  = '
    print tablePrinter.indent (labels+datatoprint, hasHeader = True, separateRows = False, prefix = '| ',
                               postfix = ' |', justify = 'right', delim = ' | ',
                               wrapfunc = lambda x: wrap_onspace_strict (x, 22))

    if len (hltpath) != 0 and hltpath != 'all':
        totalrow.append ([str (totalSelectedLS), '%.3f'% (totalRecorded), '%.3f'% (totalRecordedInPath)])
    else:
        totalrow.append ([str (totalSelectedLS), '%.3f'% (totalRecorded)])
    print ' ==  =  Total : '
    print tablePrinter.indent (lastrowlabels+totalrow, hasHeader = True, separateRows = False, prefix = '| ',
                               postfix = ' |', justify = 'right', delim = ' | ',
                               wrapfunc = lambda x: wrap_onspace (x, 20))    
    if isVerbose:
        deadtoprint = []
        deadtimelabels = [ ('Run', 'Lumi section : Dead fraction')]

        for dataperRun in lumidata:
            runnum = dataperRun[0]
            if len (dataperRun[1]) == 0:
                deadtoprint.append ([str (runnum), 'N/A'])
                continue
            perlsdata = dataperRun[2]
            #print 'perlsdata 2 : ', perlsdata
            deadT = getDeadfractions (perlsdata)
            t = ''
            for myls, de in deadT.items():
                if de<0:
                    t += str (myls)+':nobeam '
                else:
                    t += str (myls)+':'+'%.5f'% (de)+' '
            deadtoprint.append ([str (runnum), t])
        print ' ==  = '
        print tablePrinter.indent (deadtimelabels+deadtoprint, hasHeader = True, separateRows = True, prefix = '| ',
                                   postfix = ' |', justify = 'right', delim = ' | ',
                                   wrapfunc = lambda x: wrap_onspace (x, 80))


def dumpRecordedLumi (lumidata, hltpath = ''):
    #labels = ['Run', 'HLT path', 'Recorded']
    datatodump = []
    for dataperRun in lumidata:
        runnum = dataperRun[0]
        if len (dataperRun[1]) == 0:
            rowdata = []
            rowdata += [str (runnum)]+2*['N/A']
            datatodump.append (rowdata)
            continue
        perlsdata = dataperRun[2]
        recordedLumi = 0.0
        #norbits = perlsdata.values()[0][3]
        recordedLumi = calculateTotalRecorded (perlsdata)
        trgdict = dataperRun[1]
        effective = calculateEffective (trgdict, recordedLumi)
        if trgdict.has_key (hltpath) and effective.has_key (hltpath):
            rowdata = []
            l1bit = trgdict[hltpath][0]
            if len (trgdict[hltpath]) !=  3:
                rowdata += [str (runnum), hltpath, 'N/A']
            else:
                hltprescale = trgdict[hltpath][1]
                l1prescale = trgdict[hltpath][2]
                rowdata += [str (runnum), hltpath, effective[hltpath]]
            datatodump.append (rowdata)
            continue
        
        for trg, trgdata in trgdict.items():
            #print trg, trgdata
            rowdata = []                    
            rowdata += [str (runnum)]
            l1bit = trgdata[0]
            if len (trgdata) == 3:
                rowdata += [trg, effective[trg]]
            else:
                rowdata += [trg, 'N/A']
            datatodump.append (rowdata)
    return datatodump


def printOverviewData (delivered, recorded, hltpath = ''):
    if len (hltpath) == 0 or hltpath == 'all':
        toprowlabels = [ ('Run', 'Delivered LS', 'Delivered'+u' (/\u03bcb)'.encode ('utf-8'), 'Selected LS', 'Recorded'+u' (/\u03bcb)'.encode ('utf-8') )]
        lastrowlabels = [ ('Delivered LS', 'Delivered'+u' (/\u03bcb)'.encode ('utf-8'), 'Selected LS', 'Recorded'+u' (/\u03bcb)'.encode ('utf-8') ) ]
    else:
        toprowlabels = [ ('Run', 'Delivered LS', 'Delivered'+u' (/\u03bcb)'.encode ('utf-8'), 'Selected LS', 'Recorded'+u' (/\u03bcb)'.encode ('utf-8'), 'Effective'+u' (/\u03bcb) '.encode ('utf-8')+hltpath )]
        lastrowlabels = [ ('Delivered LS', 'Delivered'+u' (/\u03bcb)'.encode ('utf-8'), 'Selected LS', 'Recorded'+u' (/\u03bcb)'.encode ('utf-8'), 'Effective '+u' (/\u03bcb) '.encode ('utf-8')+hltpath)]
    datatable = []
    totaldata = []
    totalDeliveredLS = 0
    totalSelectedLS = 0
    totalDelivered = 0.0
    totalRecorded = 0.0
    totalRecordedInPath = 0.0
    totaltable = []
    for runidx, deliveredrowdata in enumerate (delivered):
        rowdata = []
        rowdata += [deliveredrowdata[0], deliveredrowdata[1], deliveredrowdata[2]]
        if deliveredrowdata[1] == 'N/A': #run does not exist
            if  hltpath != '' and hltpath != 'all':
                rowdata += ['N/A', 'N/A', 'N/A']
            else:
                rowdata += ['N/A', 'N/A']
            datatable.append (rowdata)
            continue
        totalDeliveredLS += int (deliveredrowdata[1])
        totalDelivered += float (deliveredrowdata[2])
        selectedls = recorded[runidx][2].keys()
        #print 'runidx ', runidx, deliveredrowdata
        #print 'selectedls ', selectedls
        if len (selectedls) == 0:
            selectedlsStr = '[]'
            recordedLumi = 0
            if  hltpath != '' and hltpath != 'all':
                rowdata += [selectedlsStr, 'N/A', 'N/A']
            else:
                rowdata += [selectedlsStr, 'N/A']
        else:
            selectedlsStr = splitlistToRangeString (selectedls)
            recordedLumi = calculateTotalRecorded (recorded[runidx][2])
            lumiinPaths = calculateEffective (recorded[runidx][1], recordedLumi)
            if hltpath != '' and hltpath != 'all':
                if lumiinPaths.has_key (hltpath):
                    rowdata += [selectedlsStr, '%.3f'% (recordedLumi), '%.3f'% (lumiinPaths[hltpath])]
                    totalRecordedInPath += lumiinPaths[hltpath]
                else:
                    rowdata += [selectedlsStr, '%.3f'% (recordedLumi), 'N/A']
            else:
                #rowdata += [selectedlsStr, '%.3f'% (recordedLumi), '%.3f'% (recordedLumi)]
                rowdata += [selectedlsStr, '%.3f'% (recordedLumi)]
        totalSelectedLS += len (selectedls)
        totalRecorded += recordedLumi
        datatable.append (rowdata)

    if hltpath != '' and hltpath != 'all':
        totaltable = [[str (totalDeliveredLS), '%.3f'% (totalDelivered), str (totalSelectedLS),
                       '%.3f'% (totalRecorded), '%.3f'% (totalRecordedInPath)]]
    else:
        totaltable = [[str (totalDeliveredLS), '%.3f'% (totalDelivered), str (totalSelectedLS),
                       '%.3f'% (totalRecorded)]]
    print tablePrinter.indent (toprowlabels+datatable, hasHeader = True, separateRows = False, prefix = '| ',
                               postfix = ' |', justify = 'right', delim = ' | ',
                               wrapfunc = lambda x: wrap_onspace (x, 20))
    print ' ==  =  Total : '
    print tablePrinter.indent (lastrowlabels+totaltable, hasHeader = True, separateRows = False, prefix = '| ',
                               postfix = ' |', justify = 'right', delim = ' | ',
                               wrapfunc = lambda x: wrap_onspace (x, 20))


def dumpOverview (delivered, recorded, hltpath = ''):
    #toprowlabels = ['run', 'delivered', 'recorded', 'hltpath']
    datatable = []
    for runidx, deliveredrowdata in enumerate (delivered):
        rowdata = []
        rowdata += [deliveredrowdata[0], deliveredrowdata[2]]
        if deliveredrowdata[1] == 'N/A': #run does not exist
            rowdata += ['N/A', 'N/A']
            datatable.append (rowdata)
            continue
        recordedLumi = calculateTotalRecorded (recorded[runidx][2])
        lumiinPaths = calculateEffective (recorded[runidx][1], recordedLumi)
        if hltpath != '' and hltpath != 'all':
            if lumiinPaths.has_key (hltpath):
                rowdata += [recordedLumi, lumiinPaths[hltpath]]
            else:
                rowdata += [recordedLumi, 'N/A']
        else:
            rowdata += [recordedLumi, recordedLumi]
        datatable.append (rowdata)
    return datatable


def xingLuminosityForRun (dbsession, runnum, parameters, lumiXingDict = {},
                          maxLumiSection = None):
    '''Given a run number and a minimum xing luminosity value,
    returns a dictionary (keyed by (run, lumi section)) where the
    value is a list of tuples of (xingID, xingLum).

    - For all xing luminosities, simply set minLumValue to 0.

    - If you want one dictionary for several runs, pass it in to
      "lumiXingDict"


    select 
    s.cmslsnum, d.bxlumivalue, d.bxlumierror, d.bxlumiquality, d.algoname from LUMIDETAIL d, LUMISUMMARY s where s.runnum = 133885 and d.algoname = 'OCC1' and s.lumisummary_id = d.lumisummary_id order by s.startorbit, s.cmslsnum
    '''
    try:
        runnum = int (runnum)
        dbsession.transaction().start (True)
        schema = dbsession.schema (parameters.lumischema)
        if not schema:
            raise 'cannot connect to schema ', parameters.lumischema
        detailOutput = coral.AttributeList()
        detailOutput.extend ('startorbit',    'unsigned int')
        detailOutput.extend ('cmslsnum',      'unsigned int')
        detailOutput.extend ('bxlumivalue',   'blob')
        detailOutput.extend ('bxlumierror',   'blob')
        detailOutput.extend ('bxlumiquality', 'blob')
        detailOutput.extend ('algoname',      'string')
        detailCondition = coral.AttributeList()
        detailCondition.extend ('runnum',   'unsigned int')
        detailCondition.extend ('algoname', 'string')
        detailCondition['runnum'].setData (runnum)
        detailCondition['algoname'].setData ('OCC1')
        query = schema.newQuery()
        query.addToTableList(nameDealer.lumisummaryTableName(), 's')
        query.addToTableList(nameDealer.lumidetailTableName(),  'd')
        query.addToOutputList ('s.STARTORBIT',    'startorbit')
        query.addToOutputList ('s.CMSLSNUM',      'cmslsnum')
        query.addToOutputList ('d.BXLUMIVALUE',   'bxlumivalue')
        query.addToOutputList ('d.BXLUMIERROR',   'bxlumierror')
        query.addToOutputList ('d.BXLUMIQUALITY', 'bxlumiquality')
        query.addToOutputList ('d.ALGONAME',      'algoname')
        query.setCondition ('s.RUNNUM =:runnum and d.ALGONAME =:algoname and s.LUMISUMMARY_ID=d.LUMISUMMARY_ID',detailCondition)
        query.addToOrderList ('s.CMSLSNUM')
        query.defineOutput (detailOutput)
        cursor = query.execute()
        count = 0
        while cursor.next():
            ## ## Note: If you are going to break out of this loop early,
            ## ## make sure you call cursor.close():
            ## 
            ## if count > 20 :
            ##     cursor.close()
            ##     break
            ## count  +=  1
            cmslsnum    = cursor.currentRow()['cmslsnum'].data()
            algoname    = cursor.currentRow()['algoname'].data()
            bxlumivalue = cursor.currentRow()['bxlumivalue'].data()
            startorbit  = cursor.currentRow()['startorbit'].data()
            
            if maxLumiSection and maxLumiSection < cmslsnum:
                cursor.close()
                break
            
            xingArray = array.array ('f')
            xingArray.fromstring( bxlumivalue.readline() )
            numPrinted = 0
            xingLum = []
            for index, lum in enumerate (xingArray):
                lum  *=  parameters.normFactor
                if lum < parameters.xingMinLum:
                    continue
                xingLum.append( (index, lum) )
            lumiXingDict[ (runnum, cmslsnum) ] = xingLum
        del query
        dbsession.transaction().commit()
        return lumiXingDict      
    except Exception, e:
        print str (e)
        print "whoops"
        dbsession.transaction().rollback()
        del dbsession


def flatten (obj):
    '''Given nested lists or tuples, returns a single flattened list'''
    result = []
    for piece in obj:
        if hasattr (piece, '__iter__') and not isinstance (piece, basestring):
            result.extend( flatten (piece) )
        else:
            result.append (piece)
    return result    


def mergeXingLumi (triplet, xingLumiDict):
    '''Given general xing information and a xingLumiDict, the xing
    luminosity information is merged with the general information'''
    runNumber = triplet[0]
    deadTable = triplet[2]
    for lumi, lumiList in deadTable.iteritems():
        key = ( int(runNumber), int(lumi) )
        xingLumiValues = xingLumiDict.get (key)
        if xingLumiValues:
            lumiList.append( flatten (xingLumiValues) )


def setupSession (connectString, siteconfpath, parameters, debug = False):
    '''returns database session'''
    connectparser = connectstrParser.connectstrParser (connectString)
    connectparser.parse()
    usedefaultfrontierconfig = False
    cacheconfigpath = ''
    if connectparser.needsitelocalinfo():
        if not siteconfpath:
            cacheconfigpath = os.environ['CMS_PATH']
            if cacheconfigpath:
                cacheconfigpath = os.path.join (cacheconfigpath, 'SITECONF', 'local', 'JobConfig', 'site-local-config.xml')
            else:
                usedefaultfrontierconfig = True
        else:
            cacheconfigpath = siteconfpath
            cacheconfigpath = os.path.join (cacheconfigpath, 'site-local-config.xml')
        ccp = cacheconfigParser.cacheconfigParser()
        if usedefaultfrontierconfig:
            ccp.parseString (parameters.defaultfrontierConfigString)
        else:
            ccp.parse (cacheconfigpath)
        connectString = connectparser.fullfrontierStr (connectparser.schemaname(), ccp.parameterdict())
    svc = coral.ConnectionService()
    if debug :
        msg = coral.MessageStream ('')
        msg.setMsgVerbosity (coral.message_Level_Debug)
        parameters.verbose = True
    session = svc.connect (connectString, accessMode = coral.access_ReadOnly)
    session.typeConverter().setCppTypeForSqlType ("unsigned int", "NUMBER (10)")
    session.typeConverter().setCppTypeForSqlType ("unsigned long long", "NUMBER (20)")
    return session, svc



###==============real api=====###

def allruns(schemaHandle,requireRunsummary=True,requireLumisummary=False,requireTrg=False,requireHlt=False):
    '''
    find all runs in the DB. By default requires cmsrunsummary table contain the run. The condition can be loosed in situation where db loading failed on certain data portions.
    '''
    if not requireRunsummary and not requireLumiummary and not requireTrg and not requireHlt:
        print 'must require at least one table'
        raise
    runresult=[]
    runlist=[]
    numdups=0
    if requireRunsummary:
        numdups=numdups+1
        queryHandle=schemaHandle.newQuery()
        queryHandle.addToTableList(nameDealer.cmsrunsummaryTableName())
        queryHandle.addToOutputList("RUNNUM","run")
        #queryBind=coral.AttributeList()
        result=coral.AttributeList()
        result.extend("run","unsigned int")
        queryHandle.defineOutput(result)
        cursor=queryHandle.execute()
        while cursor.next():
            r=cursor.currentRow()['run'].data()
            runlist.append(r)
        del queryHandle
    if requireLumisummary:
        numdups=numdups+1
        queryHandle=schemaHandle.newQuery()
        queryHandle.addToTableList(nameDealer.lumisummaryTableName())
        queryHandle.addToOutputList("distinct RUNNUM","run")
        #queryBind=coral.AttributeList()
        result=coral.AttributeList()
        result.extend("run","unsigned int")
        queryHandle.defineOutput(result)
        cursor=queryHandle.execute()
        while cursor.next():
            r=cursor.currentRow()['run'].data()
            runlist.append(r)
        del queryHandle
    if requireTrg:
        numdups=numdups+1
        queryHandle=schemaHandle.newQuery()
        queryHandle.addToTableList(nameDealer.trgTableName())
        queryHandle.addToOutputList("distinct RUNNUM","run")
        #queryBind=coral.AttributeList()
        result=coral.AttributeList()
        result.extend("run","unsigned int")
        queryHandle.defineOutput(result)
        cursor=queryHandle.execute()
        while cursor.next():
            r=cursor.currentRow()['run'].data()
            runlist.append(r)
        del queryHandle
    if requireHlt:
        numdups=numdups+1
        queryHandle=schemaHandle.newQuery()
        queryHandle.addToTableList(nameDealer.hltTableName())
        queryHandle.addToOutputList("distinct RUNNUM","run")
        #queryBind=coral.AttributeList()
        result=coral.AttributeList()
        result.extend("run","unsigned int")
        queryHandle.defineOutput(result)
        cursor=queryHandle.execute()
        while cursor.next():
            r=cursor.currentRow()['run'].data()
            runlist.append(r)
        del queryHandle
    dupresult=CommonUtil.count_dups(runlist)
    for dup in dupresult:
        if dup[1]==numdups:
            runresult.append(dup[0])
    runresult.sort()
    return runresult

def validation(queryHandle,run=None,cmsls=None):
    '''retrieve validation data per run or all
    input: run. if not run, retrive all; if cmslsnum selection list pesent, filter out unselected result
    output: {run:[[cmslsnum,status,comment]]}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.lumivalidationTableName())
    queryHandle.addToOutputList('RUNNUM','runnum')
    queryHandle.addToOutputList('CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('FLAG','flag')
    queryHandle.addToOutputList('COMMENT','comment')
    if run:
        queryCondition='RUNNUM=:runnum'
        queryBind=coral.AttributeList()
        queryBind.extend('runnum','unsigned int')
        queryBind['runnum'].setData(run)
        queryHandle.setCondition(queryCondition,queryBind)
    queryResult=coral.AttributeList()
    queryResult.extend('runnum','unsigned int')
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('flag','string')
    queryResult.extend('comment','string')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        runnum=cursor.currentRow()['runnum'].data()
        if not result.has_key(runnum):
            result[runnum]=[]
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        flag=cursor.currentRow()['flag'].data()
        comment=cursor.currentRow()['comment'].data()
        result[runnum].append([cmslsnum,flag,comment])
    if run and cmsls and len(cmsls)!=0:
        selectedresult={}
        for runnum,perrundata in result.items():
            for lsdata in perrundata:
                if lsdata[0] not in cmsls:
                    continue
                if not selectedresult.has_key(runnum):
                    selectedresult[runnum]=[]
                selectedresult[runnum].append(lsdata)
        return selectedresult
    else:
        return result
    
def allfills(queryHandle,filtercrazy=True):
    '''select distinct fillnum from cmsrunsummary
    there are crazy fill numbers. we assume they are not valid runs
    '''
    result=[]
    queryHandle.addToTableList(nameDealer.cmsrunsummaryTableName())
    queryHandle.addToOutputList('distinct FILLNUM','fillnum')
    
    if filtercrazy:
        queryCondition='FILLNUM>:zero and FILLNUM<:crazybig'
        queryBind=coral.AttributeList()
        queryBind.extend('zero','unsigned int')
        queryBind.extend('crazybig','unsigned int')
        queryBind['zero'].setData(int(0))
        queryBind['crazybig'].setData(int(29701))
        queryHandle.setCondition(queryCondition,queryBind)
    queryResult=coral.AttributeList()
    queryResult.extend('fillnum','unsigned int')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        result.append(cursor.currentRow()['fillnum'].data())
    result.sort()
    return result
def runsummaryByrun(queryHandle,runnum):
    '''
    select fillnum,sequence,hltkey,to_char(starttime),to_char(stoptime) from cmsrunsummary where runnum=:runnum
    output: [fillnum,sequence,hltkey,starttime,stoptime]
    '''
    t=lumiTime.lumiTime()
    result=[]
    queryHandle.addToTableList(nameDealer.cmsrunsummaryTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition['runnum'].setData(int(runnum))
    queryHandle.addToOutputList('FILLNUM','fillnum')
    queryHandle.addToOutputList('SEQUENCE','sequence')
    queryHandle.addToOutputList('HLTKEY','hltkey')
    queryHandle.addToOutputList('to_char(STARTTIME,\''+t.coraltimefm+'\')','starttime')
    queryHandle.addToOutputList('to_char(STOPTIME,\''+t.coraltimefm+'\')','stoptime')
    queryHandle.setCondition('RUNNUM=:runnum',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('fillnum','unsigned int')
    queryResult.extend('sequence','string')
    queryResult.extend('hltkey','string')
    queryResult.extend('starttime','string')
    queryResult.extend('stoptime','string')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        result.append(cursor.currentRow()['fillnum'].data())
        result.append(cursor.currentRow()['sequence'].data())
        result.append(cursor.currentRow()['hltkey'].data())
        result.append(cursor.currentRow()['starttime'].data())
        result.append(cursor.currentRow()['stoptime'].data())
    #if len(result)!=5:
    #    print 'wrong runsummary result'
    #    raise
    return result

def lumisummaryByrun(queryHandle,runnum,lumiversion,beamstatus=None,beamenergy=None,beamenergyfluctuation=0.09):
    '''
    one can impose beamstatus, beamenergy selections at the SQL query level or process them later from the general result
    select cmslsnum,instlumi,numorbit,startorbit,beamstatus,beamenery from lumisummary where runnum=:runnum and lumiversion=:lumiversion order by startorbit;
    output: [[cmslsnum,instlumi,numorbit,startorbit,beamstatus,beamenergy,cmsalive]]
    Note: the non-cmsalive LS are included in the result
    '''
    result=[]
    queryHandle.addToTableList(nameDealer.lumisummaryTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition.extend('lumiversion','string')
    conditionstring='RUNNUM=:runnum and LUMIVERSION=:lumiversion'
    queryCondition['runnum'].setData(int(runnum))
    queryCondition['lumiversion'].setData(lumiversion)
    queryHandle.addToOutputList('CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('INSTLUMI','instlumi')
    queryHandle.addToOutputList('NUMORBIT','numorbit')
    queryHandle.addToOutputList('STARTORBIT','startorbit')
    queryHandle.addToOutputList('BEAMSTATUS','beamstatus')
    queryHandle.addToOutputList('BEAMENERGY','beamenergy')
    queryHandle.addToOutputList('CMSALIVE','cmsalive')
    if beamstatus and len(beamstatus)!=0:
        conditionstring=conditionstring+' and BEAMSTATUS=:beamstatus'
        queryCondition.extend('beamstatus','string')
        queryCondition['beamstatus'].setData(beamstatus)
    if beamenergy:
        minBeamenergy=float(beamenergy*(1.0-beamenergyfluctuation))
        maxBeamenergy=float(beamenergy*(1.0+beamenergyfluctuation))
        conditionstring=conditionstring+' and BEAMENERGY>:minBeamenergy and BEAMENERGY<:maxBeamenergy'
        queryCondition.extend('minBeamenergy','float')
        queryCondition.extend('maxBeamenergy','float')
        queryCondition['minBeamenergy'].setData(float(minBeamenergy))
        queryCondition['maxBeamenergy'].setData(float(maxBeamenergy))
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('instlumi','float')
    queryResult.extend('numorbit','unsigned int')
    queryResult.extend('startorbit','unsigned int')
    queryResult.extend('beamstatus','string')
    queryResult.extend('beamenergy','float')
    queryResult.extend('cmsalive','unsigned int')
    queryHandle.defineOutput(queryResult)
    queryHandle.setCondition(conditionstring,queryCondition)
    queryHandle.addToOrderList('startorbit')
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        instlumi=cursor.currentRow()['instlumi'].data()
        numorbit=cursor.currentRow()['numorbit'].data()
        startorbit=cursor.currentRow()['startorbit'].data()
        beamstatus=cursor.currentRow()['beamstatus'].data()
        beamenergy=cursor.currentRow()['beamenergy'].data()
        cmsalive=cursor.currentRow()['cmsalive'].data()
        result.append([cmslsnum,instlumi,numorbit,startorbit,beamstatus,beamenergy,cmsalive])
    return result

def lumisumByrun(queryHandle,runnum,lumiversion,beamstatus=None,beamenergy=None,beamenergyfluctuation=0.09):
    '''
    beamenergy unit : GeV
    beamenergyfluctuation : fraction allowed to fluctuate around beamenergy value
    select sum(instlumi) from lumisummary where runnum=:runnum and lumiversion=:lumiversion
    output: float totallumi
    Note: the output is the raw result, need to apply LS length in time(sec)
    '''
    result=0.0
    queryHandle.addToTableList(nameDealer.lumisummaryTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition.extend('lumiversion','string')
    
    queryCondition['runnum'].setData(int(runnum))
    queryCondition['lumiversion'].setData(lumiversion)
    queryHandle.addToOutputList('sum(INSTLUMI)','lumitotal')
    conditionstring='RUNNUM=:runnum and LUMIVERSION=:lumiversion'
    if beamstatus and len(beamstatus)!=0:
        conditionstring=conditionstring+' and BEAMSTATUS=:beamstatus'
        queryCondition.extend('beamstatus','string')
        queryCondition['beamstatus'].setData(beamstatus)
    if beamenergy and beamenergy!=0.0:
        minBeamenergy=float(beamenergy*(1.0-beamenergyfluctuation))
        maxBeamenergy=float(beamenergy*(1.0+beamenergyfluctuation))
        conditionstring=conditionstring+' and BEAMENERGY>:minBeamenergy and BEAMENERGY<:maxBeamenergy'
        queryCondition.extend('minBeamenergy','float')
        queryCondition.extend('maxBeamenergy','float')
        queryCondition['minBeamenergy'].setData(float(minBeamenergy))
        queryCondition['maxBeamenergy'].setData(float(maxBeamenergy))
    queryHandle.setCondition(conditionstring,queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('lumitotal','float')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        result=cursor.currentRow()['lumitotal'].data()
    return result

def trgbitzeroByrun(queryHandle,runnum):
    '''
    select cmslsnum,trgcount,deadtime,bitname,prescale from trg where runnum=:runnum and bitnum=0;
    output: {cmslsnum:[trgcount,deadtime,bitname,prescale]}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.trgTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition.extend('bitnum','unsigned int')
    queryCondition['runnum'].setData(int(runnum))
    queryCondition['bitnum'].setData(int(0))
    queryHandle.addToOutputList('CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('TRGCOUNT','trgcount')
    queryHandle.addToOutputList('DEADTIME','deadtime')
    queryHandle.addToOutputList('BITNAME','bitname')
    queryHandle.addToOutputList('PRESCALE','prescale')
    queryHandle.setCondition('RUNNUM=:runnum and BITNUM=:bitnum',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('trgcount','unsigned int')
    queryResult.extend('deadtime','unsigned int')
    queryResult.extend('bitname','string')
    queryResult.extend('prescale','unsigned int')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        trgcount=cursor.currentRow()['trgcount'].data()
        deadtime=cursor.currentRow()['deadtime'].data()
        bitname=cursor.currentRow()['bitname'].data()
        prescale=cursor.currentRow()['prescale'].data()
        if not result.has_key(cmslsnum):
            result[cmslsnum]=[trgcount,deadtime,bitname,prescale]
    return result

def lumisummarytrgbitzeroByrun(queryHandle,runnum,lumiversion,beamstatus=None,beamenergy=None,beamenergyfluctuation=0.09):
    '''
    select l.cmslsnum,l.instlumi,l.numorbit,l.startorbit,l.beamstatus,l.beamenery,t.trgcount,t.deadtime,t.bitname,t.prescale from trg t,lumisummary l where t.bitnum=:bitnum and l.runnum=:runnum and l.lumiversion=:lumiversion and l.runnum=t.runnum and t.cmslsnum=l.cmslsnum; 
    Everything you ever need to know about bitzero and avg luminosity. Since we do not know if joint query is better of sperate, support both.
    output: {cmslsnum:[instlumi,numorbit,startorbit,beamstatus,beamenergy,bitzerocount,deadtime,bitname,prescale]}
    Note: only cmsalive LS are included in the result. Therefore, this function cannot be used for calculating delivered!
    '''
    result={}
    queryHandle.addToTableList(nameDealer.trgTableName(),'t')
    queryHandle.addToTableList(nameDealer.lumisummaryTableName(),'l')
    queryCondition=coral.AttributeList()
    queryCondition.extend('bitnum','unsigned int')
    queryCondition.extend('runnum','unsigned int')
    queryCondition.extend('lumiversion','string')
    queryCondition['bitnum'].setData(int(0))        
    queryCondition['runnum'].setData(int(runnum))
    queryCondition['lumiversion'].setData(lumiversion)
    
    queryHandle.addToOutputList('l.CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('l.INSTLUMI','instlumi')
    queryHandle.addToOutputList('l.NUMORBIT','numorbit')
    queryHandle.addToOutputList('l.STARTORBIT','startorbit')
    queryHandle.addToOutputList('l.BEAMSTATUS','beamstatus')
    queryHandle.addToOutputList('l.BEAMENERGY','beamenergy')
    queryHandle.addToOutputList('t.TRGCOUNT','trgcount')
    queryHandle.addToOutputList('t.DEADTIME','deadtime')
    queryHandle.addToOutputList('t.BITNAME','bitname')
    queryHandle.addToOutputList('t.PRESCALE','prescale')
    conditionstring='t.BITNUM=:bitnum and l.RUNNUM=:runnum and l.LUMIVERSION=:lumiversion and l.RUNNUM=t.RUNNUM and t.CMSLSNUM=l.CMSLSNUM'
    if beamstatus and len(beamstatus)!=0:
        conditionstring=conditionstring+' and l.BEAMSTATUS=:beamstatus'
        queryCondition.extend('beamstatus','string')
        queryCondition['beamstatus'].setData(beamstatus)
    if beamenergy and beamenergy!=0.0:
        minBeamenergy=float(beamenergy*(1-beamenergyfluctuation))
        maxBeamenergy=float(beamenergy*(1+beamenergyfluctuation))
        conditionstring=conditionstring+' and l.BEAMENERGY>:minBeamenergy and l.BEAMENERGY<:maxBeamenergy'
        queryCondition.extend('minBeamenergy','float')
        queryCondition.extend('maxBeamenergy','float')
        queryCondition['minBeamenergy'].setData(float(minBeamenergy))
        queryCondition['maxBeamenergy'].setData(float(maxBeamenergy))
    queryHandle.setCondition(conditionstring,queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('instlumi','float')
    queryResult.extend('numorbit','unsigned int')
    queryResult.extend('startorbit','unsigned int')
    queryResult.extend('beamstatus','string')
    queryResult.extend('beamenergy','float')  
    queryResult.extend('trgcount','unsigned int')
    queryResult.extend('deadtime','unsigned int')
    queryResult.extend('bitname','string')
    queryResult.extend('prescale','unsigned int')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        instlumi=cursor.currentRow()['instlumi'].data()
        numorbit=cursor.currentRow()['numorbit'].data()
        startorbit=cursor.currentRow()['startorbit'].data()
        beamstatus=cursor.currentRow()['beamstatus'].data()
        beamenergy=cursor.currentRow()['beamenergy'].data()
        trgcount=cursor.currentRow()['trgcount'].data()
        deadtime=cursor.currentRow()['deadtime'].data()
        bitname=cursor.currentRow()['bitname'].data()
        prescale=cursor.currentRow()['prescale'].data()
        if not result.has_key(cmslsnum):
            result[cmslsnum]=[instlumi,numorbit,startorbit,beamstatus,beamenergy,trgcount,deadtime,bitname,prescale]
    return result

def trgBybitnameByrun(queryHandle,runnum,bitname):
    '''
    select cmslsnum,trgcount,deadtime,bitnum,prescale from trg where runnum=:runnum and bitname=:bitname;
    output: {cmslsnum:[trgcount,deadtime,bitnum,prescale]}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.trgTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition.extend('bitname','string')
    queryCondition['runnum'].setData(int(runnum))
    queryCondition['bitname'].setData(bitname)        
    queryHandle.addToOutputList('CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('TRGCOUNT','trgcount')
    queryHandle.addToOutputList('DEADTIME','deadtime')
    queryHandle.addToOutputList('BITNUM','bitnum')
    queryHandle.addToOutputList('PRESCALE','prescale')
    queryHandle.setCondition('RUNNUM=:runnum and BITNAME=:bitname',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('trgcount','unsigned int')
    queryResult.extend('deadtime','unsigned long long')
    queryResult.extend('bitnum','unsigned int')
    queryResult.extend('prescale','unsigned int')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        trgcount=cursor.currentRow()['trgcount'].data()
        deadtime=cursor.currentRow()['deadtime'].data()
        bitnum=cursor.currentRow()['bitnum'].data()
        prescale=cursor.currentRow()['prescale'].data()
        if not result.has_key(cmslsnum):
            result[cmslsnum]=[trgcount,deadtime,bitnum,prescale]
    return result

def trgAllbitsByrun(queryHandle,runnum):
    '''
    all you ever want to know about trigger
    select cmslsnum,trgcount,deadtime,bitnum,bitname,prescale from trg where runnum=:runnum order by  bitnum,cmslsnum
    this can be changed to blob query later
    output: {cmslsnum:{bitname:[bitnum,trgcount,deadtime,prescale]}}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.trgTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition['runnum'].setData(int(runnum))
    queryHandle.addToOutputList('cmslsnum')
    queryHandle.addToOutputList('trgcount')
    queryHandle.addToOutputList('deadtime')
    queryHandle.addToOutputList('bitnum')
    queryHandle.addToOutputList('bitname')
    queryHandle.addToOutputList('prescale')
    queryHandle.setCondition('runnum=:runnum',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('trgcount','unsigned int')
    queryResult.extend('deadtime','unsigned long long')
    queryResult.extend('bitnum','unsigned int')
    queryResult.extend('bitname','string')
    queryResult.extend('prescale','unsigned int')
    queryHandle.defineOutput(queryResult)
    queryHandle.addToOrderList('bitnum')
    queryHandle.addToOrderList('cmslsnum')
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        trgcount=cursor.currentRow()['trgcount'].data()
        deadtime=cursor.currentRow()['deadtime'].data()
        bitnum=cursor.currentRow()['bitnum'].data()
        bitname=cursor.currentRow()['bitname'].data()
        prescale=cursor.currentRow()['prescale'].data()
        if not result.has_key(cmslsnum):
            dataperLS={}
            dataperLS[bitname]=[bitnum,trgcount,deadtime,prescale]
            result[cmslsnum]=dataperLS
        else:
            result[cmslsnum][bitname]=[bitnum,trgcount,deadtime,prescale]
    return result


def hltBypathByrun(queryHandle,runnum,hltpath):
    '''
    select cmslsnum,inputcount,acceptcount,prescale from hlt where runnum=:runnum and pathname=:pathname
    output: {cmslsnum:[inputcount,acceptcount,prescale]}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.hltTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition.extend('pathname','string')
    queryCondition['runnum'].setData(int(runnum))
    queryCondition['pathname'].setData(hltpath)
    queryHandle.addToOutputList('CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('INPUTCOUNT','inputcount')
    queryHandle.addToOutputList('ACCEPTCOUNT','acceptcount')
    queryHandle.addToOutputList('PRESCALE','prescale')
    queryHandle.setCondition('RUNNUM=:runnum and PATHNAME=:pathname',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('inputcount','unsigned int')
    queryResult.extend('acceptcount','unsigned int')
    queryResult.extend('prescale','unsigned int')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        inputcount=cursor.currentRow()['inputcount'].data()
        acceptcount=cursor.currentRow()['acceptcount'].data()
        prescale=cursor.currentRow()['prescale'].data()
        if not result.has_key(cmslsnum):
            result[cmslsnum]=[inputcount,acceptcount,prescale]
    return result

def hltAllpathByrun(queryHandle,runnum):
    '''
    select cmslsnum,inputcount,acceptcount,prescale,pathname from hlt where runnum=:runnum
    this can be changed to blob query later
    output: {cmslsnum:{pathname:[inputcount,acceptcount,prescale]}}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.hltTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition['runnum'].setData(int(runnum))
    queryHandle.addToOutputList('CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('INPUTCOUNT','inputcount')
    queryHandle.addToOutputList('ACCEPTCOUNT','acceptcount')
    queryHandle.addToOutputList('PRESCALE','prescale')
    queryHandle.addToOutputList('PATHNAME','pathname')
    queryHandle.setCondition('RUNNUM=:runnum',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('inputcount','unsigned int')
    queryResult.extend('acceptcount','unsigned int')
    queryResult.extend('prescale','unsigned int')
    queryResult.extend('pathname','string')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        inputcount=cursor.currentRow()['inputcount'].data()
        acceptcount=cursor.currentRow()['acceptcount'].data()
        prescale=cursor.currentRow()['prescale'].data()
        pathname=cursor.currentRow()['pathname'].data()
        if not result.has_key(cmslsnum):
            dataperLS={}
            dataperLS[pathname]=[inputcount,acceptcount,prescale]
            result[cmslsnum]=dataperLS
        else:
            result[cmslsnum][pathname]=[inputcount,acceptcount,prescale]
    return result


def beamIntensityForRun(query,parameters,runnum):
    '''
    select CMSBXINDEXBLOB,BEAMINTENSITYBLOB_1,BEAMINTENSITYBLOB_2 from LUMISUMMARY where runnum=146315 and LUMIVERSION='0001'
    
    output : result {startorbit: [(bxidx,beam1intensity,beam2intensity)]}
    '''
    result={} #{startorbit:[(bxidx,occlumi,occlumierr,beam1intensity,beam2intensity)]}
    
    lumisummaryOutput=coral.AttributeList()
    lumisummaryOutput.extend('cmslsnum','unsigned int')
    lumisummaryOutput.extend('startorbit','unsigned int')
    lumisummaryOutput.extend('bxindexblob','blob');
    lumisummaryOutput.extend('beamintensityblob1','blob');
    lumisummaryOutput.extend('beamintensityblob2','blob');
    condition=coral.AttributeList()
    condition.extend('runnum','unsigned int')
    condition.extend('lumiversion','string')
    condition['runnum'].setData(int(runnum))
    condition['lumiversion'].setData(parameters.lumiversion)
    
    query.addToTableList(parameters.lumisummaryname)
    query.addToOutputList('CMSLSNUM','cmslsnum')
    query.addToOutputList('STARTORBIT','startorbit')
    query.addToOutputList('CMSBXINDEXBLOB','bxindexblob')
    query.addToOutputList('BEAMINTENSITYBLOB_1','beamintensityblob1')
    query.addToOutputList('BEAMINTENSITYBLOB_2','beamintensityblob2')
    query.setCondition('RUNNUM=:runnum AND LUMIVERSION=:lumiversion',condition)
    query.defineOutput(lumisummaryOutput)
    cursor=query.execute()
    while cursor.next():
        #cmslsnum=cursor.currentRow()['cmslsnum'].data()
        startorbit=cursor.currentRow()['startorbit'].data()
        if not cursor.currentRow()["bxindexblob"].isNull():
            bxindexblob=cursor.currentRow()['bxindexblob'].data()
            beamintensityblob1=cursor.currentRow()['beamintensityblob1'].data()
            beamintensityblob2=cursor.currentRow()['beamintensityblob2'].data()
            if bxindexblob.readline() is not None and beamintensityblob1.readline() is not None and beamintensityblob2.readline() is not None:
                bxidx=array.array('h')
                bxidx.fromstring(bxindexblob.readline())
                bb1=array.array('f')
                bb1.fromstring(beamintensityblob1.readline())
                bb2=array.array('f')
                bb2.fromstring(beamintensityblob2.readline())
                for index,bxidxvalue in enumerate(bxidx):
                    if not result.has_key(startorbit):
                        result[startorbit]=[]
                    b1intensity=bb1[index]
                    b2intensity=bb2[index]
                    result[startorbit].append((bxidxvalue,b1intensity,b2intensity))
    return result
    
def calibratedDetailForRunLimitresult(query,parameters,runnum,algoname='OCC1'):
    '''select 
    s.cmslsnum,d.bxlumivalue,d.bxlumierror,d.bxlumiquality,d.algoname from LUMIDETAIL d,LUMISUMMARY s where s.runnum=133885 and d.algoname='OCC1' and s.lumisummary_id=d.lumisummary_id order by s.startorbit,s.cmslsnum
    result={(startorbit,cmslsnum):[(lumivalue,lumierr),]}
    '''
    result={}
    detailOutput=coral.AttributeList()
    detailOutput.extend('cmslsnum','unsigned int')
    detailOutput.extend('startorbit','unsigned int')
    detailOutput.extend('bxlumivalue','blob')
    detailOutput.extend('bxlumierror','blob')
    detailCondition=coral.AttributeList()
    detailCondition.extend('runnum','unsigned int')
    detailCondition.extend('algoname','string')
    detailCondition['runnum'].setData(runnum)
    detailCondition['algoname'].setData(algoname)

    query.addToTableList(parameters.lumisummaryname,'s')
    query.addToTableList(parameters.lumidetailname,'d')
    query.addToOutputList('s.CMSLSNUM','cmslsnum')
    query.addToOutputList('s.STARTORBIT','startorbit')
    query.addToOutputList('d.BXLUMIVALUE','bxlumivalue')
    query.addToOutputList('d.BXLUMIERROR','bxlumierror')
    query.addToOutputList('d.BXLUMIQUALITY','bxlumiquality')
    query.setCondition('s.RUNNUM=:runnum and d.ALGONAME=:algoname and s.LUMISUMMARY_ID=d.LUMISUMMARY_ID',detailCondition)
    query.defineOutput(detailOutput)
    cursor=query.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        bxlumivalue=cursor.currentRow()['bxlumivalue'].data()
        bxlumierror=cursor.currentRow()['bxlumierror'].data()
        startorbit=cursor.currentRow()['startorbit'].data()
        
        bxlumivalueArray=array.array('f')
        bxlumivalueArray.fromstring(bxlumivalue.readline())
        bxlumierrorArray=array.array('f')
        bxlumierrorArray.fromstring(bxlumierror.readline())
        xingLum=[]
        #apply selection criteria
        maxlumi=max(bxlumivalueArray)*parameters.normFactor
        for index,lum in enumerate(bxlumivalueArray):
            lum *= parameters.normFactor
            lumierror = bxlumierrorArray[index]*parameters.normFactor
            if lum<max(parameters.xingMinLum,maxlumi*0.2): 
                continue
            xingLum.append( (index,lum,lumierror) )
            if len(xingLum)!=0:
                result[(startorbit,cmslsnum)]=xingLum
    return result
   
def lumidetailByrunByAlgo(queryHandle,runnum,algoname='OCC1'):
    '''
    select s.cmslsnum,d.bxlumivalue,d.bxlumierror,d.bxlumiquality,s.startorbit from LUMIDETAIL d,LUMISUMMARY s where s.runnum=:runnum and d.algoname=:algoname and s.lumisummary_id=d.lumisummary_id order by s.startorbit
    output: [[cmslsnum,bxlumivalue,bxlumierror,bxlumiquality,startorbit]]
    since the output is ordered by time, it has to be in seq list format
    '''
    result=[]
    queryHandle.addToTableList(nameDealer.lumidetailTableName(),'d')
    queryHandle.addToTableList(nameDealer.lumisummaryTableName(),'s')
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition.extend('algoname','string')
    queryCondition['runnum'].setData(int(runnum))
    queryCondition['algoname'].setData(algoname)
    queryHandle.addToOutputList('s.CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('d.BXLUMIVALUE','bxlumivalue')
    queryHandle.addToOutputList('d.BXLUMIERROR','bxlumierror')
    queryHandle.addToOutputList('d.BXLUMIQUALITY','bxlumiquality')
    queryHandle.addToOutputList('s.STARTORBIT','startorbit')
    queryHandle.setCondition('s.runnum=:runnum and d.algoname=:algoname and s.lumisummary_id=d.lumisummary_id',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('bxlumivalue','blob')
    queryResult.extend('bxlumierror','blob')
    queryResult.extend('bxlumiquality','blob')
    queryResult.extend('startorbit','unsigned int')    
    queryHandle.addToOrderList('s.STARTORBIT')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        bxlumivalue=cursor.currentRow()['bxlumivalue'].data()
        bxlumierror=cursor.currentRow()['bxlumierror'].data()
        bxlumiquality=cursor.currentRow()['bxlumiquality'].data()
        startorbit=cursor.currentRow()['startorbit'].data()
        result.append([cmslsnum,bxlumivalue,bxlumierror,bxlumiquality,startorbit])
    return result

def lumidetailAllalgosByrun(queryHandle,runnum):
    '''
    select s.cmslsnum,d.bxlumivalue,d.bxlumierror,d.bxlumiquality,d.algoname,s.startorbit from LUMIDETAIL d,LUMISUMMARY s where s.runnum=:runnumber and s.lumisummary_id=d.lumisummary_id order by s.startorbit,d.algoname
    output: {algoname:{cmslsnum:[bxlumivalue,bxlumierror,bxlumiquality,startorbit]}}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.lumidetailTableName(),'d')
    queryHandle.addToTableList(nameDealer.lumisummaryTableName(),'s')
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition['runnum'].setData(int(runnum))
    queryHandle.addToOutputList('s.CMSLSNUM','cmslsnum')
    queryHandle.addToOutputList('d.BXLUMIVALUE','bxlumivalue')
    queryHandle.addToOutputList('d.BXLUMIERROR','bxlumierror')
    queryHandle.addToOutputList('d.BXLUMIQUALITY','bxlumiquality')
    queryHandle.addToOutputList('d.ALGONAME','algoname')
    queryHandle.addToOutputList('s.STARTORBIT','startorbit')
    queryHandle.setCondition('s.RUNNUM=:runnum and s.LUMISUMMARY_ID=d.LUMISUMMARY_ID',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('cmslsnum','unsigned int')
    queryResult.extend('bxlumivalue','blob')
    queryResult.extend('bxlumierror','blob')
    queryResult.extend('bxlumiquality','blob')
    queryResult.extend('algoname','string')
    queryResult.extend('startorbit','unsigned int')    
    queryHandle.addToOrderList('startorbit')
    queryHandle.addToOrderList('algoname')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        cmslsnum=cursor.currentRow()['cmslsnum'].data()
        bxlumivalue=cursor.currentRow()['bxlumivalue'].data()
        bxlumierror=cursor.currentRow()['bxlumierror'].data()
        bxlumiquality=cursor.currentRow()['bxlumiquality'].data()
        algoname=cursor.currentRow()['algoname'].data()
        startorbit=cursor.currentRow()['startorbit'].data()
        if not result.has_key(algoname):
            dataPerAlgo={}
            dataPerAlgo[cmslsnum]=[bxlumivalue,bxlumierror,bxlumiquality,startorbit]
            result[algoname]=dataPerAlgo
        else:
            result[algoname][cmslsnum]=[bxlumivalue,bxlumierror,bxlumiquality,startorbit]           
    return result

def hlttrgMappingByrun(queryHandle,runnum):
    '''
    select m.hltpathname,m.l1seed from cmsrunsummary r,trghltmap m where r.runnum=:runnum and m.hltkey=r.hltkey
    output: {hltpath:l1seed}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.cmsrunsummaryTableName(),'r')
    queryHandle.addToTableList(nameDealer.trghltMapTableName(),'m')
    queryCondition=coral.AttributeList()
    queryCondition.extend('runnum','unsigned int')
    queryCondition['runnum'].setData(int(runnum))
    queryHandle.addToOutputList('m.HLTPATHNAME','hltpathname')
    queryHandle.addToOutputList('m.L1SEED','l1seed')
    queryHandle.setCondition('r.RUNNUM=:runnum and m.HLTKEY=r.HLTKEY',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('hltpathname','string')
    queryResult.extend('l1seed','string')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        hltpathname=cursor.currentRow()['hltpathname'].data()
        l1seed=cursor.currentRow()['l1seed'].data()
        if not result.has_key(hltpathname):
            result[hltpathname]=l1seed
    return result

def runsByfillrange(queryHandle,minFill,maxFill):
    '''
    find all runs in the fill range inclusive
    select runnum,fillnum from cmsrunsummary where fillnum>=:minFill and fillnum<=:maxFill
    output: fillDict={fillnum:[runlist]}
    '''
    result={}
    queryHandle.addToTableList(nameDealer.cmsrunsummaryTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('minFill','unsigned int')
    queryCondition.extend('maxFill','unsigned int')
    queryCondition['minFill'].setData(int(minFill))
    queryCondition['maxFill'].setData(int(maxFill))
    queryHandle.addToOutputList('RUNNUM','runnum')
    queryHandle.addToOutputList('FILLNUM','fillnum')
    queryHandle.setCondition('FILLNUM>=:minFill and FILLNUM<=:maxFill',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('runnum','unsigned int')
    queryResult.extend('fillnum','unsigned int')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        runnum=cursor.currentRow()['runnum'].data()
        fillnum=cursor.currentRow()['fillnum'].data()
        if not result.has_key(fillnum):
            result[fillnum]=[runnum]
        else:
            result[fillnum].append(runnum)
    return result

def runsByTimerange(queryHandle,minTime,maxTime):
    '''
    find all runs in the time range inclusive
    the selected run must have started after minTime and finished by maxTime
    select runnum,to_char(startTime),to_char(stopTime) from cmsrunsummary where startTime>=timestamp(minTime) and stopTime<=timestamp(maxTime);
    input: minTime,maxTime in python obj datetime.datetime
    output: {runnum:[starttime,stoptime]} return in python obj datetime.datetime
    '''
    t=lumiTime.lumiTime()
    result={}
    coralminTime=coral.TimeStamp(minTime.year,minTime.month,minTime.day,minTime.hour,minTime.minute,minTime.second,0)
    coralmaxTime=coral.TimeStamp(maxTime.year,maxTime.month,maxTime.day,maxTime.hour,maxTime.minute,maxTime.second,0)
    queryHandle.addToTableList(nameDealer.cmsrunsummaryTableName())
    queryCondition=coral.AttributeList()
    queryCondition.extend('minTime','time stamp')
    queryCondition.extend('maxTime','time stamp')
    queryCondition['minTime'].setData(coralminTime)
    queryCondition['maxTime'].setData(coralmaxTime)
    queryHandle.addToOutputList('RUNNUM','runnum')
    queryHandle.addToOutputList('TO_CHAR(STARTTIME,\''+t.coraltimefm+'\')','starttime')
    queryHandle.addToOutputList('TO_CHAR(STOPTIME,\''+t.coraltimefm+'\')','stoptime')
    queryHandle.setCondition('STARTTIME>=:minTime and STOPTIME<=:maxTime',queryCondition)
    queryResult=coral.AttributeList()
    queryResult.extend('runnum','unsigned int')
    queryResult.extend('starttime','string')
    queryResult.extend('stoptime','string')
    queryHandle.defineOutput(queryResult)
    cursor=queryHandle.execute()
    while cursor.next():
        runnum=cursor.currentRow()['runnum'].data()
        starttimeStr=cursor.currentRow()['starttime'].data()
        stoptimeStr=cursor.currentRow()['stoptime'].data()
        if not result.has_key(runnum):
            result[runnum]=[t.StrToDatetime(starttimeStr),t.StrToDatetime(stoptimeStr)]
    return result
    
if __name__=='__main__':
    msg=coral.MessageStream('')
    #msg.setMsgVerbosity(coral.message_Level_Debug)
    msg.setMsgVerbosity(coral.message_Level_Error)
    os.environ['CORAL_AUTH_PATH']='/afs/cern.ch/cms/DB/lumi'
    svc = coral.ConnectionService()
    connectstr='oracle://cms_orcoff_prod/cms_lumi_prod'
    session=svc.connect(connectstr,accessMode=coral.access_ReadOnly)
    session.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")
    session.typeConverter().setCppTypeForSqlType("unsigned long long","NUMBER(20)")
    session.transaction().start(True)
    schema=session.nominalSchema()
    allruns=allruns(schema,requireLumisummary=True,requireTrg=True,requireHlt=True)
    print 'allruns in runsummary and lumisummary and trg and hlt ',len(allruns)
    #q=schema.newQuery()
    #runsummaryOut=runsummaryByrun(q,139400)
    #del q
    #q=schema.newQuery()
    #lumisummaryOut=lumisummaryByrun(q,139400,'0001')
    #del q
    #q=schema.newQuery()
    #lumisummaryOutStablebeam7TeV=lumisummaryByrun(q,139400,'0001',beamstatus='STABLE BEAMS',beamenergy=3.5E003,beamenergyfluctuation=0.09)
    #del q
    #q=schema.newQuery()
    #lumitotal=lumisumByrun(q,139400,'0001')
    #del q
    #q=schema.newQuery()
    #lumitotalStablebeam7TeV=lumisumByrun(q,139400,'0001',beamstatus='STABLE BEAMS',beamenergy=3.5E003,beamenergyfluctuation=0.09)
    #del q
    #q=schema.newQuery()
    #trgbitzero=trgbitzeroByrun(q,139400)
    #del q
    #q=schema.newQuery()
    #lumijointrg=lumisummarytrgbitzeroByrun(q,135525,'0001')
    #del q
    #q=schema.newQuery()
    #lumijointrgStablebeam7TeV=lumisummarytrgbitzeroByrun(q,135525,'0001',beamstatus='STABLE BEAMS',beamenergy=3.5E003,beamenergyfluctuation=0.09)
    #del q
    #q=schema.newQuery()
    #trgforbit=trgBybitnameByrun(q,139400,'L1_ZeroBias')
    #del q
    #q=schema.newQuery()
    #trgallbits=trgAllbitsByrun(q,139400)
    #del q
    #q=schema.newQuery()
    #hltbypath=hltBypathByrun(q,139400,'HLT_Mu5')
    #del q
    #q=schema.newQuery()
    #hltallpath=hltAllpathByrun(q,139400)
    #del q
    #q=schema.newQuery()
    #hlttrgmap=hlttrgMappingByrun(q,139400)
    #del q
    #q=schema.newQuery()
    #occ1detail=lumidetailByrunByAlgo(q,139400,'OCC1')
    #del q
    #q=schema.newQuery()
    #alldetail=lumidetailAllalgosByrun(q,139400)
    #del q
    #q=schema.newQuery()
    #runsbyfill=runsByfillrange(q,1150,1170)
    #del q
    #now=datetime.datetime.now()
    #aweek=datetime.timedelta(weeks=1)
    #lastweek=now-aweek
    #print lastweek
    #q=schema.newQuery()
    #runsinaweek=runsByTimerange(q,lastweek,now)
    #del q
    q=schema.newQuery()
    allfills=allfills(q)
    del q
    session.transaction().commit()  
    del session
    del svc
    #print 'runsummaryByrun : ',runsummaryOut
    #print
    #print 'lumisummaryByrun : ',lumisummaryOut
    #print '######'
    #print 'lumisummaryByrun stable beams 7TeV : ',lumisummaryOutStablebeam7TeV
    #print '######'
    #print 'totallumi : ',lumitotal
    #print
    #print
    #print 'totallumi stable beam and 7TeV: ',lumitotalStablebeam7TeV
    #print
    #print 'trgbitzero : ',trgbitzero
    #print 
    #print 'lumijointrg : ', lumijointrg
    #print 'total LS : ',len(lumijointrg)
    #print 'lumijointrg stable beams 7TeV :', lumijointrgStablebeam7TeV
    #print 'total LS : ',len(lumijointrgStablebeam7TeV)
    #print 'trgforbit L1_ZeroBias ',trgforbit
    #print
    #print 'trgallbits ',trgallbits[1] #big query. be aware of speed
    #print
    #print 'hltforpath HLT_Mu5',hltbypath
    #print
    #print 'hltallpath ',hltallpath
    #print
    #print 'hlttrgmap ',hlttrgmap
    #print
    #print 'lumidetail occ1 ',len(occ1detail)
    #print
    #print 'runsbyfill ',runsbyfill
    #print
    #print 'runsinaweek ',runsinaweek.keys()
    print 'all fills ',allfills
