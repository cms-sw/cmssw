#!/usr/bin/env python
VERSION='2.00'
import os,sys
import coral
from RecoLuminosity.LumiDB import argparse,nameDealer,selectionParser,hltTrgSeedMapper,connectstrParser,cacheconfigParser,tablePrinter,csvReporter,csvSelectionParser
from RecoLuminosity.LumiDB.wordWrappers import wrap_always,wrap_onspace,wrap_onspace_strict
class constants(object):
    def __init__(self):
        self.NORM=1.0
        self.LUMIVERSION='0001'
        self.NBX=3564
        self.BEAMMODE='stable' #possible choices stable,quiet,either
        self.VERBOSE=False
    def defaultfrontierConfigString(self):
        return """<frontier-connect><proxy url="http://cmst0frontier.cern.ch:3128"/><proxy url="http://cmst0frontier.cern.ch:3128"/><proxy url="http://cmst0frontier1.cern.ch:3128"/><proxy url="http://cmst0frontier2.cern.ch:3128"/><server url="http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier1.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier2.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier3.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier4.cern.ch:8000/FrontierInt"/></frontier-connect>"""
    
def lslengthsec(numorbit,numbx):
    #print numorbit, numbx
    l=numorbit*numbx*25e-09
    return l

def lsBylsLumi(deadtable):
    """
    input: {lsnum:[deadtime,instlumi,bit_0,norbits]}
    output: {lsnum:[instlumi,recordedlumi]}
    """
    result={}
    for myls,d in deadtable.items():
        lstime=lslengthsec(d[3],3564)
        instlumi=d[1]*lstime
        if float(d[2])==0.0:
            deadfrac=1.0
        else:
            deadfrac=float(d[0])/float(d[2])
        recordedLumi=instlumi*(1.0-deadfrac)
        result[myls]=[instlumi,recordedLumi]
    return result
def deliveredLumiForRun(dbsession,c,runnum):
    #
    #select sum(INSTLUMI),count(INSTLUMI) from lumisummary where runnum=124025 and lumiversion='0001';
    #apply norm factor and ls length in sec on the query result 
    #unit E27cm^-2 
    #
    #if c.VERBOSE:
    #    print 'deliveredLumiForRun : norm : ',c.NORM,' : run : ',runnum
    #output ['run','totalls','delivered','beammode']
    delivered=0.0
    totalls=0
    try:
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        query=schema.tableHandle(nameDealer.lumisummaryTableName()).newQuery()
        query.addToOutputList("sum(INSTLUMI)","totallumi")
        query.addToOutputList("count(INSTLUMI)","totalls")
        query.addToOutputList("NUMORBIT","norbits")
        queryBind=coral.AttributeList()
        queryBind.extend("runnum","unsigned int")
        queryBind.extend("lumiversion","string")
        queryBind["runnum"].setData(int(runnum))
        queryBind["lumiversion"].setData(c.LUMIVERSION)
        result=coral.AttributeList()
        result.extend("totallumi","float")
        result.extend("totalls","unsigned int")
        result.extend("norbits","unsigned int")
        query.defineOutput(result)
        query.setCondition("RUNNUM =:runnum AND LUMIVERSION =:lumiversion",queryBind)
        query.limitReturnedRows(1)
        query.groupBy('NUMORBIT')
        cursor=query.execute()
        while cursor.next():
            delivereddata=cursor.currentRow()['totallumi'].data()
            totallsdata=cursor.currentRow()['totalls'].data()
            norbitsdata=cursor.currentRow()['norbits'].data()
            if delivereddata:
                totalls=totallsdata
                norbits=norbitsdata
                lstime=lslengthsec(norbits,c.NBX)
                delivered=delivereddata*c.NORM*lstime
        del query
        dbsession.transaction().commit()
        lumidata=[]

        if delivered==0.0:
            lumidata=[str(runnum),'N/A','N/A','N/A']
        else:
            lumidata=[str(runnum),str(totalls),'%.3f'%delivered,c.BEAMMODE]
        return lumidata
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession

def deliveredLumiForRange(dbsession,c,fileparsingResult):
    #
    #in this case,only take run numbers from theinput file
    #
    lumidata=[]
    runs= fileparsingResult.runs()
    runs.sort()
    for run in runs:
        lumidata.append( deliveredLumiForRun(dbsession,c,run) )
    return lumidata

def recordedLumiForRun(dbsession,c,runnum,lslist=[]):
    """output: ['runnumber','trgtable{}','deadtable{}']
    """
    recorded=0.0
    lumidata=[] #[runnumber,trgtable,deadtable]
    trgtable={} #{hltpath:[l1seed,hltprescale,l1prescale]}
    deadtable={} #{lsnum:[deadtime,instlumi,bit_0,norbits]}
    lumidata.append(runnum)
    lumidata.append(trgtable)
    lumidata.append(deadtable)
    collectedseeds=[] #[(hltpath,l1seed)]
    try:
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        query=schema.newQuery()
        query.addToTableList(nameDealer.cmsrunsummaryTableName(),'cmsrunsummary')
        query.addToTableList(nameDealer.trghltMapTableName(),'trghltmap')#small table first
        queryCondition=coral.AttributeList()
        queryCondition.extend("runnumber","unsigned int")
        queryCondition["runnumber"].setData(int(runnum))
        query.setCondition("trghltmap.HLTKEY=cmsrunsummary.HLTKEY AND cmsrunsummary.RUNNUM=:runnumber",queryCondition)
        query.addToOutputList("trghltmap.HLTPATHNAME","hltpathname")
        query.addToOutputList("trghltmap.L1SEED","l1seed")
        result=coral.AttributeList()
        result.extend("hltpathname","string")
        result.extend("l1seed","string")
        query.defineOutput(result)
        cursor=query.execute()
        while cursor.next():
            hltpathname=cursor.currentRow()["hltpathname"].data()
            l1seed=cursor.currentRow()["l1seed"].data()
            collectedseeds.append((hltpathname,l1seed))
        #print 'collectedseeds ',collectedseeds
        del query
        dbsession.transaction().commit()
        #loop over hltpath
        for (hname,sname) in collectedseeds:
            l1bitname=hltTrgSeedMapper.findUniqueSeed(hname,sname)
            #print 'found unque seed ',hname,l1bitname
            if l1bitname:
                lumidata[1][hname]=[]
                lumidata[1][hname].append(l1bitname.replace('\"',''))
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        hltprescQuery=schema.tableHandle(nameDealer.hltTableName()).newQuery()
        hltprescQuery.addToOutputList("PATHNAME","hltpath")
        hltprescQuery.addToOutputList("PRESCALE","hltprescale")
        hltprescCondition=coral.AttributeList()
        hltprescCondition.extend('runnumber','unsigned int')
        hltprescCondition.extend('cmslsnum','unsigned int')
        hltprescCondition.extend('inf','unsigned int')
        hltprescResult=coral.AttributeList()
        hltprescResult.extend('hltpath','string')
        hltprescResult.extend('hltprescale','unsigned int')
        hltprescQuery.defineOutput(hltprescResult)
        hltprescCondition['runnumber'].setData(int(runnum))
        hltprescCondition['cmslsnum'].setData(1)
        hltprescCondition['inf'].setData(0)
        hltprescQuery.setCondition("RUNNUM =:runnumber and CMSLSNUM =:cmslsnum and PRESCALE !=:inf",hltprescCondition)
        cursor=hltprescQuery.execute()
        while cursor.next():
            hltpath=cursor.currentRow()['hltpath'].data()
            hltprescale=cursor.currentRow()['hltprescale'].data()
            if lumidata[1].has_key(hltpath):
                lumidata[1][hltpath].append(hltprescale)
                
        cursor.close()
        del hltprescQuery
        dbsession.transaction().commit()
        
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        query=schema.newQuery()
        query.addToTableList(nameDealer.trgTableName(),'trg')
        query.addToTableList(nameDealer.lumisummaryTableName(),'lumisummary')#small table first--right-most
        queryCondition=coral.AttributeList()
        queryCondition.extend("runnumber","unsigned int")
        queryCondition.extend("lumiversion","string")
        #queryCondition.extend("alive","bool")
        queryCondition["runnumber"].setData(int(runnum))
        queryCondition["lumiversion"].setData(c.LUMIVERSION)
        #queryCondition["alive"].setData(True)
        query.setCondition("lumisummary.RUNNUM=:runnumber and lumisummary.LUMIVERSION =:lumiversion AND lumisummary.CMSLSNUM=trg.CMSLSNUM and lumisummary.RUNNUM=trg.RUNNUM",queryCondition)
        #query.setCondition("trg.RUNNUM =:runnumber AND lumisummary.RUNNUM=:runnumber and lumisummary.LUMIVERSION =:lumiversion AND lumisummary.CMSLSNUM=trg.CMSLSNUM AND lumisummary.cmsalive=:alive AND trg.BITNUM=:bitnum",queryCondition)
        #query.addToOutputList("sum(lumisummary.INSTLUMI*(1-trg.DEADTIME/(lumisummary.numorbit*3564)))","recorded")
        query.addToOutputList("lumisummary.CMSLSNUM","cmsls")
        query.addToOutputList("lumisummary.INSTLUMI","instlumi")
        query.addToOutputList("lumisummary.NUMORBIT","norbits")
        query.addToOutputList("trg.TRGCOUNT","trgcount")
        query.addToOutputList("trg.BITNAME","bitname")
        query.addToOutputList("trg.DEADTIME","trgdeadtime")
        query.addToOutputList("trg.PRESCALE","trgprescale")
        query.addToOutputList("trg.BITNUM","trgbitnum")
        #query.addToOrderList("trg.BITNAME")
        #query.addToOrderList("trg.CMSLSNUM")

        result=coral.AttributeList()
        result.extend("cmsls","unsigned int")
        result.extend("instlumi","float")
        result.extend("norbits","unsigned int")
        result.extend("trgcount","unsigned int")
        result.extend("bitname","string")
        result.extend("trgdeadtime","unsigned long long")
        result.extend("trgprescale","unsigned int")
        result.extend("trgbitnum","unsigned int")
        trgprescalemap={}
        query.defineOutput(result)
        cursor=query.execute()
        while cursor.next():
            cmsls=cursor.currentRow()["cmsls"].data()
            instlumi=cursor.currentRow()["instlumi"].data()*c.NORM
            norbits=cursor.currentRow()["norbits"].data()
            trgcount=cursor.currentRow()["trgcount"].data()
            trgbitname=cursor.currentRow()["bitname"].data()
            trgdeadtime=cursor.currentRow()["trgdeadtime"].data()
            trgprescale=cursor.currentRow()["trgprescale"].data()
            trgbitnum=cursor.currentRow()["trgbitnum"].data()
            if cmsls==1:
                if not trgprescalemap.has_key(trgbitname):
                    trgprescalemap[trgbitname]=trgprescale
            if trgbitnum==0:
                if not deadtable.has_key(cmsls):
                    deadtable[cmsls]=[]
                    deadtable[cmsls].append(trgdeadtime)
                    deadtable[cmsls].append(instlumi)
                    deadtable[cmsls].append(trgcount)
                    deadtable[cmsls].append(norbits)
        cursor.close()
        del query
        dbsession.transaction().commit()
        
        #
        #consolidate results
        #
        #trgtable
        #print 'trgprescalemap',trgprescalemap
        #print lumidata[1]
        for hpath,trgdataseq in lumidata[1].items():   
            bitn=trgdataseq[0]
            if trgprescalemap.has_key(bitn) and len(trgdataseq)==2:
                lumidata[1][hpath].append(trgprescalemap[bitn])                
        #filter selected cmsls
        lumidata[2]=filterDeadtable(deadtable,lslist)
        #print 'lumidata[2] ',lumidata[2]
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
    #print 'before return lumidata ',lumidata
    return lumidata

def filterDeadtable(inTable,lslist):
    result={}
    if len(lslist)==0: #if request no ls, then return nothing
        #return inTable
        return result
    for existingLS in inTable.keys():
        if existingLS in lslist:
            result[existingLS]=inTable[existingLS]
    return result

def recordedLumiForRange(dbsession,c,fileparsingResult):
    #
    #in this case,only take run numbers from theinput file
    #
    lumidata=[]
    runs=fileparsingResult.runs()
    runs.sort()
    runsandls=fileparsingResult.runsandls()
    ###print 'runsandls ',runsandls
    for run in runs:
        lslist=runsandls[run]
    #for (run,lslist) in fileparsingResult.runsandls().items():
        #print 'processing run ',run
        #print 'valid ls list ',lslist
        lumidata.append( recordedLumiForRun(dbsession,c,run,lslist) )
    return lumidata

def printDeliveredLumi(lumidata,mode):
    labels=[('Run','Delivered LS','Delivered'+u' (/\u03bcb)'.encode('utf-8'),'Beam Mode')]
    print tablePrinter.indent(labels+lumidata,hasHeader=True,separateRows=False,prefix='| ',postfix=' |',justify='right',delim=' | ',wrapfunc=lambda x: wrap_onspace(x,20) )

def dumpData(lumidata,filename):
    """
    input params: lumidata [{'fieldname':value}]
                  filename csvname
    """
    
    r=csvReporter.csvReporter(filename)
    r.writeRows(lumidata)

def calculateTotalRecorded(deadtable):
    """
    input: {lsnum:[deadtime,instlumi,bit_0,norbits]}
    output: recordedLumi
    """
    recordedLumi=0.0
    for myls,d in deadtable.items():
        instLumi=d[1]
        #deadfrac=float(d[0])/float(d[2]*3564)
        #print myls,float(d[2])
        if float(d[2])==0.0:
            deadfrac=1.0
        else:
            deadfrac=float(d[0])/float(d[2])
        lstime=lslengthsec(d[3],3564)
        recordedLumi+=instLumi*(1.0-deadfrac)*lstime
    return recordedLumi

def splitlistToRangeString(inPut):
    result=[]
    first=inPut[0]
    last=inPut[0]
    result.append([inPut[0]])
    counter=0
    for i in inPut[1:]:
        if i==last+1:
            result[counter].append(i)
        else:
            counter+=1
            result.append([i])
        last=i
    return ' '.join(['['+str(min(x))+'-'+str(max(x))+']' for x in result])

def calculateEffective(trgtable,totalrecorded):
    """
    input: trgtable{hltpath:[l1seed,hltprescale,l1prescale]},totalrecorded(float)
    output:{hltpath,recorded}
    """
    #print 'inputtrgtable',trgtable
    result={}
    for hltpath,data in trgtable.items():
        if len(data) == 3:
            result[hltpath]=totalrecorded/(data[1]*data[2])
        else:
            result[hltpath]=0.0
    return result

def getDeadfractions(deadtable):
    """
    inputtable: {lsnum:[deadtime,instlumi,bit_0,norbits]}
    output: {lsnum:deadfraction}
    """
    result={}
    for myls,d in deadtable.items():
        #deadfrac=float(d[0])/(float(d[2])*float(3564))
        if float(d[2])==0.0: ##no beam
            deadfrac=-1.0
        else:
            deadfrac=float(d[0])/(float(d[2]))
        result[myls]=deadfrac
    return result

def printPerLSLumi(lumidata,isVerbose=False,hltpath=''):
    '''
    input lumidata  [['runnumber','trgtable{}','deadtable{}']]
    deadtable {lsnum:[deadtime,instlumi,bit_0,norbits]}
    '''
    print 'input ',lumidata
    datatoprint=[]
    totalrow=[]
    labels=[('Run','LS','Delivered','Recorded'+u' (/\u03bcb)'.encode('utf-8'))]
    lastrowlabels=[('Selected LS','Delivered'+u' (/\u03bcb)'.encode('utf-8'),'Recorded'+u' (/\u03bcb)'.encode('utf-8'))]
    totalDeliveredLS=0
    totalSelectedLS=0
    totalDelivered=0.0
    totalRecorded=0.0
    
    for perrundata in lumidata:
        runnumber=perrundata[0]
        deadtable=perrundata[2]
        lumiresult=lsBylsLumi(deadtable)
        totalSelectedLS=totalSelectedLS+len(deadtable)
        for lsnum,dataperls in lumiresult.items():
            rowdata=[]
            if len(dataperls)==0:
                rowdata+=[str(runnumber),str(lsnum),'N/A','N/A']
            else:
                rowdata+=[str(runnumber),str(lsnum),'%.3f'%(dataperls[0]),'%.3f'%(dataperls[1])]
                totalDelivered=totalDelivered+dataperls[0]
                totalRecorded=totalRecorded+dataperls[1]
            datatoprint.append(rowdata)
    totalrow.append([str(totalSelectedLS),'%.3f'%(totalDelivered),'%.3f'%(totalRecorded)])
    print datatoprint
    print '==='
    print tablePrinter.indent(labels+datatoprint,hasHeader=True,separateRows=False,prefix='| ',postfix=' |',justify='right',delim=' | ',wrapfunc=lambda x: wrap_onspace_strict(x,22))
    print '=== Total : '
    print tablePrinter.indent(lastrowlabels+totalrow,hasHeader=True,separateRows=False,prefix='| ',postfix=' |',justify='right',delim=' | ',wrapfunc=lambda x: wrap_onspace(x,20))    
def dumpPerLSLumi(lumidata,hltpath=''):
    datatodump=[]
    for perrundata in lumidata:
        runnumber=perrundata[0]
        deadtable=perrundata[2]
        lumiresult=lsBylsLumi(deadtable)
        for lsnum,dataperls in lumiresult.items():
            rowdata=[]
            if len(dataperls)==0:
                rowdata+=[str(runnumber),str(lsnum),'N/A','N/A']
            else:
                rowdata+=[str(runnumber),str(lsnum),dataperls[0],dataperls[1]]
            datatodump.append(rowdata)
    return datatodump
def printRecordedLumi(lumidata,isVerbose=False,hltpath=''):
    datatoprint=[]
    totalrow=[]
    labels=[('Run','HLT path','Recorded'+u' (/\u03bcb)'.encode('utf-8'))]
    lastrowlabels=[('Selected LS','Recorded'+u' (/\u03bcb)'.encode('utf-8'))]
    if len(hltpath)!=0 and hltpath!='all':
        lastrowlabels=[('Selected LS','Recorded'+u' (/\u03bcb)'.encode('utf-8'),'Effective '+u'(/\u03bcb) '.encode('utf-8')+hltpath)]
    if isVerbose:
        labels=[('Run','HLT-path','L1-bit','L1-presc','HLT-presc','Recorded'+u' (/\u03bcb)'.encode('utf-8'))]
    totalSelectedLS=0
    totalRecorded=0.0
    totalRecordedInPath=0.0
    
    for dataperRun in lumidata:
        runnum=dataperRun[0]
        if len(dataperRun[1])==0:
            rowdata=[]
            rowdata+=[str(runnum)]+2*['N/A']
            datatoprint.append(rowdata)
            continue
        perlsdata=dataperRun[2]
        totalSelectedLS=totalSelectedLS+len(perlsdata)
        recordedLumi=0.0
        #norbits=perlsdata.values()[0][3]
        recordedLumi=calculateTotalRecorded(perlsdata)
        totalRecorded=totalRecorded+recordedLumi
        trgdict=dataperRun[1]
        effective=calculateEffective(trgdict,recordedLumi)
        if trgdict.has_key(hltpath) and effective.has_key(hltpath):
            rowdata=[]
            l1bit=trgdict[hltpath][0]
            if len(trgdict[hltpath]) != 3:
                if not isVerbose:
                    rowdata+=[str(runnum),hltpath,'N/A']
                else:
                    rowdata+=[str(runnum),hltpath,l1bit,'N/A','N/A','N/A']
            else:
                if not isVerbose:
                    rowdata+=[str(runnum),hltpath,'%.3f'%(effective[hltpath])]
                else:
                    hltprescale=trgdict[hltpath][1]
                    l1prescale=trgdict[hltpath][2]
                    rowdata+=[str(runnum),hltpath,l1bit,str(l1prescale),str(hltprescale),'%.3f'%(effective[hltpath])]
                totalRecordedInPath=totalRecordedInPath+effective[hltpath]
            datatoprint.append(rowdata)
            continue
        
        for trg,trgdata in trgdict.items():
            #print trg,trgdata
            rowdata=[]                    
            if trg==trgdict.keys()[0]:
                rowdata+=[str(runnum)]
            else:
                rowdata+=['']
            l1bit=trgdata[0]
            if len(trgdata)==3:
                if not isVerbose:
                    rowdata+=[trg,'%.3f'%(effective[trg])]
                else:
                    hltprescale=trgdata[1]
                    l1prescale=trgdata[2]
                    rowdata+=[trg,l1bit,str(l1prescale),str(hltprescale),'%.3f'%(effective[trg])]
            else:
                if not isVerbose:
                    rowdata+=[trg,'N/A']
                else:
                    rowdata+=[trg,l1bit,'N/A','N/A','%.3f'%(effective[trg])]
            datatoprint.append(rowdata)
    #print datatoprint
    print '==='
    print tablePrinter.indent(labels+datatoprint,hasHeader=True,separateRows=False,prefix='| ',postfix=' |',justify='right',delim=' | ',wrapfunc=lambda x: wrap_onspace_strict(x,22))

    if len(hltpath)!=0 and hltpath!='all':
        totalrow.append([str(totalSelectedLS),'%.3f'%(totalRecorded),'%.3f'%(totalRecordedInPath)])
    else:
        totalrow.append([str(totalSelectedLS),'%.3f'%(totalRecorded)])
    print '=== Total : '
    print tablePrinter.indent(lastrowlabels+totalrow,hasHeader=True,separateRows=False,prefix='| ',postfix=' |',justify='right',delim=' | ',wrapfunc=lambda x: wrap_onspace(x,20))    
    if isVerbose:
        deadtoprint=[]
        deadtimelabels=[('Run','Lumi section : Dead fraction')]

        for dataperRun in lumidata:
            runnum=dataperRun[0]
            if len(dataperRun[1])==0:
                deadtoprint.append([str(runnum),'N/A'])
                continue
            perlsdata=dataperRun[2]
            #print 'perlsdata 2 : ',perlsdata
            deadT=getDeadfractions(perlsdata)
            t=''
            for myls,de in deadT.items():
                if de<0:
                    t+=str(myls)+':nobeam '
                else:
                    t+=str(myls)+':'+'%.5f'%(de)+' '
            deadtoprint.append([str(runnum),t])
        print '==='
        print tablePrinter.indent(deadtimelabels+deadtoprint,hasHeader=True,separateRows=True,prefix='| ',postfix=' |',justify='right',delim=' | ',wrapfunc=lambda x: wrap_onspace(x,80))
        



def dumpRecordedLumi(lumidata,hltpath=''):
    #labels=['Run','HLT path','Recorded']
    datatodump=[]
    for dataperRun in lumidata:
        runnum=dataperRun[0]
        if len(dataperRun[1])==0:
            rowdata=[]
            rowdata+=[str(runnum)]+2*['N/A']
            datatodump.append(rowdata)
            continue
        perlsdata=dataperRun[2]
        recordedLumi=0.0
        #norbits=perlsdata.values()[0][3]
        recordedLumi=calculateTotalRecorded(perlsdata)
        trgdict=dataperRun[1]
        effective=calculateEffective(trgdict,recordedLumi)
        if trgdict.has_key(hltpath) and effective.has_key(hltpath):
            rowdata=[]
            l1bit=trgdict[hltpath][0]
            if len(trgdict[hltpath]) != 3:
                rowdata+=[str(runnum),hltpath,'N/A']
            else:
                hltprescale=trgdict[hltpath][1]
                l1prescale=trgdict[hltpath][2]
                rowdata+=[str(runnum),hltpath,effective[hltpath]]
            datatodump.append(rowdata)
            continue
        
        for trg,trgdata in trgdict.items():
            #print trg,trgdata
            rowdata=[]                    
            rowdata+=[str(runnum)]
            l1bit=trgdata[0]
            if len(trgdata)==3:
                rowdata+=[trg,effective[trg]]
            else:
                rowdata+=[trg,'N/A']
            datatodump.append(rowdata)
    return datatodump
def printOverviewData(delivered,recorded,hltpath=''):
    if len(hltpath)==0 or hltpath=='all':
        toprowlabels=[('Run','Delivered LS','Delivered'+u'(/\u03bcb)'.encode('utf-8'),'Selected LS','Recorded'+u'(/\u03bcb)'.encode('utf-8') )]
        lastrowlabels=[('Delivered LS','Delivered'+u' (/\u03bcb)'.encode('utf-8'),'Selected LS','Recorded'+u'(/\u03bcb)'.encode('utf-8') ) ]
    else:
        toprowlabels=[('Run','Delivered LS','Delivered'+u'(/\u03bcb)'.encode('utf-8'),'Selected LS','Recorded'+u'(/\u03bcb)'.encode('utf-8'),'Effective'+u'(/\u03bcb) '.encode('utf-8')+hltpath )]
        lastrowlabels=[('Delivered LS','Delivered'+u'(/\u03bcb)'.encode('utf-8'),'Selected LS','Recorded'+u'(/\u03bcb)'.encode('utf-8'),'Effective '+u'(/\u03bcb) '.encode('utf-8')+hltpath)]
    datatable=[]
    totaldata=[]
    totalDeliveredLS=0
    totalSelectedLS=0
    totalDelivered=0.0
    totalRecorded=0.0
    totalRecordedInPath=0.0
    totaltable=[]
    for runidx,deliveredrowdata in enumerate(delivered):
        rowdata=[]
        rowdata+=[deliveredrowdata[0],deliveredrowdata[1],deliveredrowdata[2]]
        if deliveredrowdata[1]=='N/A': #run does not exist
            if  hltpath!='' and hltpath!='all':
                rowdata+=['N/A','N/A','N/A']
            else:
                rowdata+=['N/A','N/A']
            datatable.append(rowdata)
            continue
        totalDeliveredLS+=int(deliveredrowdata[1])
        totalDelivered+=float(deliveredrowdata[2])
        
        selectedls=recorded[runidx][2].keys()
        #print 'runidx ',runidx,deliveredrowdata
        #print 'selectedls ',selectedls
        if len(selectedls)==0:
            selectedlsStr='[]'
            if  hltpath!='' and hltpath!='all':
                rowdata+=[selectedlsStr,'N/A','N/A']
            else:
                rowdata+=[selectedlsStr,'N/A']
        else:
            selectedlsStr=splitlistToRangeString(selectedls)
            recordedLumi=calculateTotalRecorded(recorded[runidx][2])
            lumiinPaths=calculateEffective(recorded[runidx][1],recordedLumi)
            if hltpath!='' and hltpath!='all':
                if lumiinPaths.has_key(hltpath):
                    rowdata+=[selectedlsStr,'%.3f'%(recordedLumi),'%.3f'%(lumiinPaths[hltpath])]
                    totalRecordedInPath+=lumiinPaths[hltpath]
                else:
                    rowdata+=[selectedlsStr,'%.3f'%(recordedLumi),'N/A']
            else:
                #rowdata+=[selectedlsStr,'%.3f'%(recordedLumi),'%.3f'%(recordedLumi)]
                rowdata+=[selectedlsStr,'%.3f'%(recordedLumi)]
        totalSelectedLS+=len(selectedls)
        totalRecorded+=recordedLumi
        datatable.append(rowdata)
    if hltpath!='' and hltpath!='all':
        totaltable=[[str(totalDeliveredLS),'%.3f'%(totalDelivered),str(totalSelectedLS),'%.3f'%(totalRecorded),'%.3f'%(totalRecordedInPath)]]
    else:
        totaltable=[[str(totalDeliveredLS),'%.3f'%(totalDelivered),str(totalSelectedLS),'%.3f'%(totalRecorded)]]
    print tablePrinter.indent(toprowlabels+datatable,hasHeader=True,separateRows=False,prefix='| ',postfix=' |',justify='right',delim=' | ',wrapfunc=lambda x: wrap_onspace(x,20))
    print '=== Total : '
    print tablePrinter.indent(lastrowlabels+totaltable,hasHeader=True,separateRows=False,prefix='| ',postfix=' |',justify='right',delim=' | ',wrapfunc=lambda x: wrap_onspace(x,20))


def dumpOverview(delivered,recorded,hltpath=''):
    #toprowlabels=['run','delivered','recorded','hltpath']
    datatable=[]
    for runidx,deliveredrowdata in enumerate(delivered):
        rowdata=[]
        rowdata+=[deliveredrowdata[0],deliveredrowdata[2]]
        if deliveredrowdata[1]=='N/A': #run does not exist
            rowdata+=['N/A','N/A']
            datatable.append(rowdata)
            continue
        recordedLumi=calculateTotalRecorded(recorded[runidx][2])
        lumiinPaths=calculateEffective(recorded[runidx][1],recordedLumi)
        if hltpath!='' and hltpath!='all':
            if lumiinPaths.has_key(hltpath):
                rowdata+=[recordedLumi,lumiinPaths[hltpath]]
            else:
                rowdata+=[recordedLumi,'N/A']
        else:
            rowdata+=[recordedLumi,recordedLumi]
        datatable.append(rowdata)
    return datatable
def main():
    c=constants()
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Lumi Calculations")
    # add required arguments
    parser.add_argument('-c',dest='connect',action='store',required=False,help='connect string to lumiDB (optional, default to frontier://LumiProd/CMS_LUMI_PROD)')
    # add optional arguments
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file')
    parser.add_argument('-n',dest='normfactor',action='store',help='normalization factor (optional, default to 1.0)')
    parser.add_argument('-r',dest='runnumber',action='store',help='run number')
    parser.add_argument('-i',dest='inputfile',action='store',help='lumi range selection file (optional)')
    parser.add_argument('-o',dest='outputfile',action='store',help='output to csv file (optional)')
    parser.add_argument('-b',dest='beammode',action='store',help='beam mode, optional for delivered action, default "stable", choices "stable","quiet","either"')
    parser.add_argument('-lumiversion',dest='lumiversion',action='store',help='lumi data version, optional for all, default 0001')
    parser.add_argument('-hltpath',dest='hltpath',action='store',help='specific hltpath to calculate the recorded luminosity, default to all')
    parser.add_argument('-siteconfpath',dest='siteconfpath',action='store',help='specific path to site-local-config.xml file, default to $CMS_PATH/SITECONF/local/JobConfig, if path undefined, fallback to cern proxy&server')
    parser.add_argument('action',choices=['overview','delivered','recorded','lumibyls'],help='command actions')
    parser.add_argument('--verbose',dest='verbose',action='store_true',help='verbose mode for printing' )
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    # parse arguments
    args=parser.parse_args()
    connectstring='frontier://LumiProd/CMS_LUMI_PROD'
    if args.connect: 
        connectstring=args.connect
    connectparser=connectstrParser.connectstrParser(connectstring)
    connectparser.parse()
    usedefaultfrontierconfig=False
    cacheconfigpath=''
    if connectparser.needsitelocalinfo():
        if not args.siteconfpath:
            cacheconfigpath=os.environ['CMS_PATH']
            if cacheconfigpath:
                cacheconfigpath=os.path.join(cacheconfigpath,'SITECONF','local','JobConfig','site-local-config.xml')
            else:
                usedefaultfrontierconfig=True
        else:
            cacheconfigpath=args.siteconfpath
            cacheconfigpath=os.path.join(cacheconfigpath,'site-local-config.xml')
        p=cacheconfigParser.cacheconfigParser()
        if usedefaultfrontierconfig:
            p.parseString(c.defaultfrontierConfigString)
        else:
            p.parse(cacheconfigpath)
        connectstring=connectparser.fullfrontierStr(connectparser.schemaname(),p.parameterdict())
    #print 'connectstring',connectstring
    runnumber=0
    svc = coral.ConnectionService()
    isverbose=False
    if args.debug :
        msg=coral.MessageStream('')
        msg.setMsgVerbosity(coral.message_Level_Debug)
        c.VERBOSE=True
    hpath=''
    ifilename=''
    ofilename=''
    beammode='stable'
    if args.verbose :
        c.VERBOSE=True
    if args.authpath and len(args.authpath)!=0:
        os.environ['CORAL_AUTH_PATH']=args.authpath
    if args.normfactor:
        c.NORM=float(args.normfactor)
    if args.lumiversion:
        c.LUMIVERSION=args.lumiversion
    if args.beammode:
        c.BEAMMODE=args.beammode
    if args.inputfile and len(args.inputfile)!=0:
        ifilename=args.inputfile
    if args.runnumber :
        runnumber=args.runnumber
    if args.outputfile and len(args.outputfile)!=0:
        ofilename=args.outputfile
    if len(ifilename)==0 and runnumber==0:
        raise "must specify either a run (-r) or an input run selection file (-i)"
    session=svc.connect(connectstring,accessMode=coral.access_Update)
    session.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")
    session.typeConverter().setCppTypeForSqlType("unsigned long long","NUMBER(20)")
    inputfilecontent=''
    fileparsingResult=''
    if runnumber==0 and len(ifilename)!=0 :
        basename,extension=os.path.splitext(ifilename)
        if extension=='.csv':#if file ends with .csv,use csv parser,else parse as json file
            fileparsingResult=csvSelectionParser.csvSelectionParser(ifilename)
        else:
            f=open(ifilename,'r')
            inputfilecontent=f.read()
            fileparsingResult=selectionParser.selectionParser(inputfilecontent)
        if not fileparsingResult:
            print 'failed to parse the input file',ifilename
            raise 
    lumidata=[]
    if args.action == 'delivered':
        if runnumber!=0:
            lumidata.append(deliveredLumiForRun(session,c,runnumber))
        else:
            lumidata=deliveredLumiForRange(session,c,fileparsingResult)    
        if not ofilename:
            printDeliveredLumi(lumidata,'')
        else:
            lumidata.insert(0,['run','nls','delivered','beammode'])
            dumpData(lumidata,ofilename)
    if args.action == 'recorded':
        if args.hltpath and len(args.hltpath)!=0:
            hpath=args.hltpath
        if runnumber!=0:
            lumidata.append(recordedLumiForRun(session,c,runnumber))
        else:
            lumidata=recordedLumiForRange(session,c,fileparsingResult)
        if not ofilename:
            printRecordedLumi(lumidata,c.VERBOSE,hpath)
        else:
            todump=dumpRecordedLumi(lumidata,hpath)
            todump.insert(0,['run','hltpath','recorded'])
            dumpData(todump,ofilename)
    if args.action == 'overview':
        delivereddata=[]
        recordeddata=[]
        if args.hltpath and len(args.hltpath)!=0:
            hpath=args.hltpath
        if runnumber!=0:
            delivereddata.append(deliveredLumiForRun(session,c,runnumber))
            recordeddata.append(recordedLumiForRun(session,c,runnumber))
        else:
            delivereddata=deliveredLumiForRange(session,c,fileparsingResult)
            recordeddata=recordedLumiForRange(session,c,fileparsingResult)
        if not ofilename:
            printOverviewData(delivereddata,recordeddata,hpath)
        else:
            todump=dumpOverview(delivereddata,recordeddata,hpath)
            if len(hpath)==0:
                hpath='all'
            todump.insert(0,['run','delivered','recorded','hltpath:'+hpath])
            dumpData(todump,ofilename)
    if args.action == 'lumibyls':
        recordeddata=[]
        if runnumber!=0:
            recordeddata.append(recordedLumiForRun(session,c,runnumber))
        else:
            recordeddata=recordedLumiForRange(session,c,fileparsingResult)
        #print 'recordedata ',recordeddata
        if not ofilename:
            printPerLSLumi(recordeddata,c.VERBOSE,hpath)
        else:
            todump=dumpPerLSLumi(recordeddata,hpath)
            if len(hpath)==0:
                hpath='all'
            todump.insert(0,['run','ls','delivered','recorded'])
            dumpData(todump,ofilename)
    #print lumidata
    
    del session
    del svc
if __name__=='__main__':
    main()
    
