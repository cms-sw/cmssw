#!/usr/bin/env python
VERSION='2.00'
import os,sys,time
import coral
#import optparse
from RecoLuminosity.LumiDB import lumiTime,inputFilesetParser,csvSelectionParser, selectionParser,csvReporter,argparse,CommonUtil,lumiQueryAPI
#import RecoLuminosity.LumiDB.lumiQueryAPI as LumiQueryAPI
#from pprint import pprint

def getPerLSData(dbsession,inputRange,lumiversion='0001'):
    result={}#{run:[[cmslsnum,orbittime,orbittimestamp,delivered,recorded]]}
    datacollector={}
    if isinstance(inputRange, str):
        datacollector[int(inputRange)]=[]
    else:
        for run in inputRange.runs():
            datacollector[run]=[]
    try:
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        for run in datacollector.keys():
            runsummaryOut=[]  #[fillnum,sequence,hltkey,starttime,stoptime]
            lumisummaryOut=[] #[[cmslsnum,instlumi,numorbit,startorbit,beamstatus,beamenergy,cmsalive]]
            trgOut={} #{cmslsnum:[trgcount,deadtime,bitname,prescale]}
            q=schema.newQuery()
            runsummaryOut=lumiQueryAPI.runsummaryByrun(q,run)
            del q
            q=schema.newQuery()
            lumisummaryOut=lumiQueryAPI.lumisummaryByrun(q,run,lumiversion)
            del q
            q=schema.newQuery()
            trgOut=lumiQueryAPI.trgbitzeroByrun(q,run,)
            del q
            if len(runsummaryOut)!=0 and len(lumisummaryOut)!=0 and len(trgOut)!=0:
                datacollector[run].append(runsummaryOut)
                datacollector[run].append(lumisummaryOut)
                datacollector[run].append(trgOut)
        dbsession.transaction().commit()
    except Exception, e:
        dbsession.transaction().rollback()
        del dbsession
        raise Exception, 'lumiCalc.getPerLSData:'+str(e)
    for run,perrundata in datacollector.items():
        result[run]=[]
        if len(perrundata)==0:
            continue
        runsummary=perrundata[0]
        lumisummary=perrundata[1]
        trg=perrundata[2]
        starttimestr=runsummaryOut[3]
        t=lumiTime.lumiTime()
        for dataperls in lumisummary:
            cmslsnum=dataperls[0]
            instlumi=dataperls[1]
            numorbit=dataperls[2]
            dellumi=instlumi*float(numorbit)*3564.0*25.0e-09
            startorbit=dataperls[3]
            orbittime=t.OrbitToTime(starttimestr,startorbit)
            orbittimestamp=time.mktime(orbittime.timetuple())+orbittime.microsecond/1e6
            trgcount=0
            deadtime=0
            prescale=0
            deadfrac=1.0
            if trg.has_key(cmslsnum):
                trgcount=trg[cmslsnum][0]
                deadtime=trg[cmslsnum][1]
                prescale=trg[cmslsnum][3]
                if trgcount!=0 and prescale!=0:
                    deadfrac=float(deadtime)/(float(trgcount)*float(prescale))
                recordedlumi=dellumi*(1.0-deadfrac)
            result[run].append( [cmslsnum,orbittime,orbittimestamp,dellumi,recordedlumi] )
    return result

def getValidationData(dbsession,run=None,cmsls=None):
    '''retrieve validation data per run or all
    input: runnum, if not runnum, retrive all
    output: {run:[[cmslsnum,flag,comment]]}
    '''
    try:
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        queryHandle=dbsession.nominalSchema().newQuery()
        result=lumiQueryAPI.validation(queryHandle,run,cmsls)
        del queryHandle
        dbsession.transaction().commit()
    except Exception, e:
        dbsession.transaction().rollback()
        del dbsession
        raise Exception, 'lumiValidate.getValidationData:'+str(e)
    return result

##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Lumi Calculations",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    allowedActions = ['overview', 'delivered', 'recorded', 'lumibyls','lumibylstime','lumibylsXing','status']
    beamModeChoices = [ "stable", "quiet", "either"]
    # parse arguments
    parser.add_argument('action',choices=allowedActions,help='command actions')
    parser.add_argument('-c',dest='connect',action='store',required=False,help='connect string to lumiDB,optional',default='frontier://LumiCalc/CMS_LUMI_PROD')
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file,optional')
    parser.add_argument('-n',dest='normfactor',action='store',type=float,default=1.0,help='normalization factor,optional')
    parser.add_argument('-r',dest='runnumber',action='store',type=int,help='run number,optional')
    parser.add_argument('-i',dest='inputfile',action='store',help='lumi range selection file,optional')
    parser.add_argument('-o',dest='outputfile',action='store',help='output to csv file,optional')
    parser.add_argument('-b',dest='beammode',action='store',choices=beamModeChoices,required=False,help='beam mode choice')
    parser.add_argument('-lumiversion',dest='lumiversion',action='store',default='0001',help='lumi data version, optional')
    parser.add_argument('-hltpath',dest='hltpath',action='store',default='all',help='specific hltpath to calculate the recorded luminosity,optional')
    parser.add_argument('-siteconfpath',dest='siteconfpath',action='store',help='specific path to site-local-config.xml file, optional. If path undefined, fallback to cern proxy&server')
    parser.add_argument('-xingMinLum', dest = 'xingMinLum', type=float, default=1e-3,required=False,help='Minimum luminosity considered for "lumibylsXing" action')
    parser.add_argument('--verbose',dest='verbose',action='store_true',help='verbose mode for printing' )
    parser.add_argument('--nowarning',dest='nowarning',action='store_true',help='suppress bad for lumi warnings' )
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    options=parser.parse_args()
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath
    ## Let's start the fun
    if not options.inputfile and not options.runnumber:
        print "must specify either a run (-r) or an input run selection file (-i)"
        sys.exit()

    ## Save what we need in the parameters object
    parameters = lumiQueryAPI.ParametersObject()
    parameters.verbose     = options.verbose
    parameters.noWarnings  = options.nowarning
    parameters.norm        = options.normfactor
    parameters.lumiversion = options.lumiversion
    if options.beammode=='stable':
        parameters.beammode    = 'STABLE BEAMS'
    parameters.xingMinLum  = options.xingMinLum
    session,svc =  lumiQueryAPI.setupSession (options.connect or \
                                              'frontier://LumiCalc/CMS_LUMI_PROD',
                                               options.siteconfpath,parameters,options.debug)
    lumiXing = False
    if options.action in ['lumibylsXing','delivered','recorded','overview','lumibyls','lumibylstime']:
        if options.action == 'lumibylsXing':
           #action = 'lumibyls'
           parameters.lumiXing = True
           # we can't have lumiXing mode if we're not writing to a CSV
           # file
           #if not options.outputfile:
           #    raise RuntimeError, "You must specify an outputt file in 'lumibylsXing' mode"
        if options.runnumber:
            inputRange=str(options.runnumber)
        else:
            inputRange=inputFilesetParser.inputFilesetParser(options.inputfile)
        if not inputRange:
            print 'failed to parse the input file', options.inputfile
            raise 

        # Delivered
        if options.action ==  'delivered':
            lumidata=lumiQueryAPI.deliveredLumiForRange(session, parameters, inputRange)    
            if not options.outputfile:
                lumiQueryAPI.printDeliveredLumi (lumidata, '')
            else:
                lumidata.insert (0, ['run', 'nls', 'delivered', 'beammode'])
                lumiQueryAPI.dumpData (lumidata, options.outputfile)

        # Recorded
        if options.action ==  'recorded':
            hltpath = ''
            if options.hltpath:
                hltpath = options.hltpath
                lumidata =  lumiQueryAPI.recordedLumiForRange (session, parameters, inputRange)
            if not options.outputfile:
                lumiQueryAPI.printRecordedLumi (lumidata, parameters.verbose, hltpath)
            else:
                todump = lumiQueryAPI.dumpRecordedLumi (lumidata, hltpath)
                todump.insert (0, ['run', 'hltpath', 'recorded'])
                lumiQueryAPI.dumpData (todump, options.outputfile)
                
                # Overview
        if options.action ==  'overview':
            hltpath=''
            if options.hltpath:
                hltpath=options.hltpath
            delivereddata=lumiQueryAPI.deliveredLumiForRange(session, parameters, inputRange)
            recordeddata=lumiQueryAPI.recordedLumiForRange(session, parameters, inputRange)
            if not options.outputfile:
                lumiQueryAPI.printOverviewData (delivereddata, recordeddata, hltpath)
            else:
                todump = lumiQueryAPI.dumpOverview (delivereddata, recordeddata, hltpath)
                if not hltpath:
                    hltpath = 'all'
                todump.insert (0, ['run', 'delivered', 'recorded', 'hltpath:'+hltpath])
                lumiQueryAPI.dumpData (todump, options.outputfile)

                # Lumi by lumisection
        if options.action == 'lumibylstime':
            lsdata=getPerLSData(session,inputRange)#{run:[[ls,orbittime,orbittimestamp,delivered,recorded],[]]}
            runs=lsdata.keys()
            runs.sort()
            if not options.outputfile:
                print 'run,cmslsnum, utctime, unixtimestamp, delivered, recorded'
                for run in runs:
                    if len(lsdata[run])==0:continue #empty or non-existing run
                    for perlsdata in lsdata[run]:
                        if len(perlsdata)==0:
                            continue
                        print run,perlsdata[0],perlsdata[1],perlsdata[2],perlsdata[3],perlsdata[4]
                            
            else:
                report=csvReporter.csvReporter(options.outputfile)
                report.writeRow(['run','cmslsnum','utctime','unixtimestamp','delivered','recorded'])
                for run in runs:
                    if len(lsdata[run])==0:continue #empty or non-existing run
                    for perlsdata in lsdata[run]:
                        if len(perlsdata)==0:
                            continue
                        report.writeRow([run,perlsdata[0],perlsdata[1],perlsdata[2],perlsdata[3],perlsdata[4]])
                            
        if options.action=='lumibyls' or options.action=='lumibylsXing':
            recordeddata=lumiQueryAPI.recordedLumiForRange(session, parameters, inputRange)
            # we got it, now we got to decide what to do with it
            if not options.outputfile:
                lumiQueryAPI.printPerLSLumi (recordeddata, parameters.verbose)
            else:
                todump = lumiQueryAPI.dumpPerLSLumi(recordeddata)
                todump.insert (0, ['run', 'ls', 'delivered', 'recorded'])
                lumiQueryAPI.dumpData (todump, options.outputfile)
        if not options.nowarning:
            result={}
            if isinstance(inputRange,str):
                result=getValidationData(session,run=int(inputRange))
            else:
                runsandls=inputRange.runsandls()
                for runnum,lslist in runsandls.items():
                    dataperrun=getValidationData(session,run=runnum,cmsls=lslist)
                    if dataperrun.has_key(runnum):
                        result[runnum]=dataperrun[runnum]
            for run,perrundata in result.items():
                totalsuspect=0
                totalbad=0
                totalunknown=0
                for lsdata in perrundata:
                    if lsdata[1]=='UNKNOWN':
                        totalunknown+=1
                    if lsdata[1]=='SUSPECT':
                        totalsuspect+=1
                    if lsdata[1]=='BAD':
                        totalbad+=1
                if totalsuspect!=0 or totalbad!=0 or totalunknown!=0:
                    print '[WARNING] : run '+str(run)+' : total non-GOOD LS: UNKNOWN '+str(totalunknown)+', BAD '+str(totalbad)+', SUSPECT '+str(totalsuspect)
    # relate to validation status
    elif options.action=='status':
        result={}
        if options.inputfile:
            p=inputFilesetParser.inputFilesetParser(options.inputfile)
            runsandls=p.runsandls()
            for runnum,lslist in runsandls.items():
                dataperrun=getValidationData(session,run=runnum,cmsls=lslist)
                result[runnum]=dataperrun[runnum]
        else:
            result=getValidationData(session,run=options.runnumber)
        runs=result.keys()
        runs.sort()
        if options.outputfile:
            r=csvReporter.csvReporter(options.outputfile)
            for run in runs:
                for perrundata in result[run]:
                    r.writeRow([str(run),str(perrundata[0]),perrundata[1],perrundata[2]])
        else:
            for run in runs:
                print '== ='
                for lsdata in result[run]:
                    print str(run)+','+str(lsdata[0])+','+lsdata[1]+','+lsdata[2]
    del session
    del svc 
