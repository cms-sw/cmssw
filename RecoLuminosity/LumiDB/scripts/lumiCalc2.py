#!/usr/bin/env python
VERSION='2.00'
import os,sys,time
import coral
#import optparse
from RecoLuminosity.LumiDB import sessionManager,lumiTime,inputFilesetParser,csvSelectionParser,selectionParser,csvReporter,argparse,CommonUtil,lumiCalcAPI,lumiReport

beamChoices=['PROTPHYS','IONPHYS']

def getDeliveredPerLS(dbsession,inputRange,amodetag='PROTPHYS',beamstatus=None,beamenergy=None,beamenergyFluc=0.2,withBXInfo=False,bxAlgo='OCC1',xingMinLum=1.0e-4,withBeamInfo=False,normname=None,datatag=None):
    '''
    input:
    output:#{run:[[lumilsnum,timestr,timestamp,delivered,(bxvalueblob,bxerrblob),(bxidx,b1intensity,b2intensity)],[]]}
    '''
    pass

def getOverviewPerLS(dbsession,inputRange,amodetag='PROTPHYS',beamstatus=None,beamenergy=None,beamenergyFluc=0.2,withBXInfo=False,bxAlgo='OCC1',xingMinLum=1.0e-4,withBeamInfo=False,normname=None,datatag=None):
    '''
    input:
    output:
        {run:[[lumilsnum,cmslsnum,timestr,timestamp,delivered,recorded,(bxvalueblob,bxerrblob),(bxidx,b1intensity,b2intensity)],[]]}
    '''
    result={}
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

def getEffectivePerLS(dbsession,inputRange,hltpathname=None,hltpathpattern=None,amodetag='PROTPHYS',beamstatus=None,beamenergy=None,beamenergyFluc=0.2,withBXInfo=False,xingMinLum=1.0e-4,bxAlgo='OCC1',withBeamInfo=False,normname=None,datatag=None):
    '''
    output:
    {run:[[lumilsnum,cmslsnum,timestr,timestamp,delivered,recorded,effdict,(bxvalueblob,bxerrblob),(bxidx,b1intensity,b2intensity)],[]]}
    '''
    pass

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
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Lumi Calculation",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    allowedActions = ['overview', 'delivered', 'recorded', 'lumibyls','lumibylsXing','status']
    beamModeChoices = [ "stable", "quiet", "either"]
    amodetagChoices = [ "PROTPHYS","IONPHYS" ]
    #
    # parse arguments
    #  
    #
    # basic arguments
    #
    parser.add_argument('action',choices=allowedActions,help='command actions')
    parser.add_argument('-c',dest='connect',action='store',required=False,help='connect string to lumiDB,optional',default='frontier://LumiCalc/CMS_LUMI_PROD')
    parser.add_argument('-P',dest='authpath',action='store',required=False,help='path to authentication file (optional)')
    parser.add_argument('-r',dest='runnumber',action='store',type=int,required=False,help='run number (optional)')
    parser.add_argument('-o',dest='outputfile',action='store',required=False,help='output to csv file (optional)')
    #
    #optional arg to select exact run and ls
    #
    parser.add_argument('-i',dest='inputfile',action='store',required=False,help='lumi range selection file (optional)')
    #
    #optional arg to select exact hltpath or pattern
    #
    parser.add_argument('-hltpath',dest='hltpath',action='store',default=None,required=False,help='specific hltpath or hltpath pattern to calculate the effectived luminosity (optional)')
    #
    #optional args to filter *runs*, they do not select on LS level.
    #
    parser.add_argument('-b',dest='beammode',action='store',choices=beamModeChoices,required=False,help='beam mode choices [stable] (optional)')
    parser.add_argument('-fill',dest='fillnum',action='store',default=None,required=False,help='fill number (optional) ')
    parser.add_argument('-amodetag',dest='amodetag',action='store',choices=amodetagChoices,required=False,help='specific accelerator mode choices [PROTOPHYS,IONPHYS] (optional)')
    parser.add_argument('-beamenergy',dest='beamenergy',action='store',type=float,default=None,help='nominal beam energy in GeV')
    parser.add_argument('-beamfluctuation',dest='beamfluctuation',type=float,action='store',default=0.02,required=False,help='fluctuation in fraction allowed to nominal beam energy, default 0.02, to be used together with -beamenergy  (optional)')
    parser.add_argument('-begin',dest='begin',action='store',default=None,required=False,help='run selection begin time, mm/dd/yy hh:mm:ss.00 (optional)')
    parser.add_argument('-end',dest='end',action='store',default=None,required=False,help='run selection stop time, mm/dd/yy hh:mm:ss.00 (optional)')    
    #
    #optional args to filter ls
    #
    parser.add_argument('-xingMinLum', dest = 'xingMinLum', type=float,default=1e-03,required=False,help='Minimum luminosity considered for lumibylsXing action')
    #
    #optional args for data and normalization version control
    #
    parser.add_argument('-lumiversion',dest='lumiversion',action='store',default=None,required=False,help='data version, optional')
    parser.add_argument('-n',dest='normfactor',action='store',default=None,required=False,help='use specify the name or the value of the normalization to use,optional')
    #
    #command configuration 
    #
    parser.add_argument('-siteconfpath',dest='siteconfpath',action='store',default=None,required=False,help='specific path to site-local-config.xml file, optional. If path undefined, fallback to cern proxy&server')
    #
    #switches
    #
    parser.add_argument('--verbose',dest='verbose',action='store_true',help='verbose mode for printing' )
    parser.add_argument('--nowarning',dest='nowarning',action='store_true',help='suppress bad for lumi warnings' )
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    
    options=parser.parse_args()
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath
        
    pbeammode = None
    normfactor=None
    if options.beammode=='stable':
        pbeammode    = 'STABLE BEAMS'
    if options.verbose:
        print 'General configuration'
        print '\tconnect: ',options.connect
        print '\tauthpath: ',options.authpath
        print '\tlumi data version: ',options.lumiversion
        print '\tsiteconfpath: ',options.siteconfpath
        print '\toutputfile: ',options.outputfile
        if options.action=='recorded' and options.hltpath:
            print 'Action: effective luminosity in hltpath: ',options.hltpath
        else:
            print 'Action: ',options.action
        if options.normfactor:
            normfactor=options.normfactor
            if CommonUtil.is_floatstr(normfactor):
                print '\tuse norm factor value ',normfactor                
            else:
                print '\tuse specific norm factor name ',normfactor
        else:
            print '\tuse norm factor in context (amodetag,beamenergy)'
        if options.runnumber: # if runnumber specified, do not go through other run selection criteria
            print '\tselect specific run== ',options.runnumber
        else:
            print '\trun selections == '
            print '\tinput selection file: ',options.inputfile
            print '\tbeam mode: ',options.beammode
            print '\tfill: ',options.fillnum
            print '\tamodetag: ',options.amodetag
            print '\tbegin: ',options.begin
            print '\tend: ',options.end
            print '\tbeamenergy: ',options.beamenergy 
            if options.beamenergy:
                print '\tbeam energy: ',str(options.beamenergy)+'+/-'+str(options.beamfluctuation*options.beamenergy)+'(GeV)'
        if options.action=='lumibylsXing':
            print '\tLS filter for lumibylsXing xingMinLum: ',options.xingMinLum
        
    svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,debugON=options.debug)
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    
    irunlsdict={}
    if options.runnumber: # if runnumber specified, do not go through other run selection criteria
        irunlsdict[options.runnumber]=None
    else:
        reqTrg=False
        reqHlt=False
        if options.action!='delivered' and  options.action!='status':
            reqTrg=True
            if options.action=='recorded':
                reqHlt=True
        session.transaction().start(True)
        schema=session.nominalSchema()
        runlist=lumiCalcAPI.runList(schema,options.fillnum,runmin=None,runmax=None,startT=options.begin,stopT=options.end,l1keyPattern=None,hltkeyPattern=None,amodetag=options.amodetag,nominalEnergy=options.beamenergy,energyFlut=options.beamfluctuation,requiretrg=reqTrg,requirehlt=reqHlt)
        session.transaction().commit()
        if options.inputfile:
            p=inputFilesetParser.inputFilesetParser(options.inputfile)
            runlsbyfile=p.runsandls()
            for runinfile in sorted(runlsbyfile):
                if runinfile not in runlist:
                    irunlsdict[runinfile]=None
                    continue
                irunlsdict[runinfile]=runlsbyfile[runinfile]
        else:
            for run in runlist:
                irunlsdict[run]=None
    if options.verbose:
        print 'Selected run:ls'
        for run in sorted(irunlsdict):
            if irunlsdict[run] is not None:
                print '\t%d : %s'%(run,','.join([str(ls) for ls in irunlsdict[run]]))
            else:
                print '\t%d : all'%run
    if options.action == 'delivered':
        session.transaction().start(True)
        result=lumiCalcAPI.deliveredLumiForRange(session.nominalSchema(),irunlsdict,beamstatus=pbeammode,norm=normfactor)
        session.transaction().commit()
        if not options.outputfile:
            lumiReport.toScreenTotDelivered(result,options.verbose)
        else:
            lumiReport.toCSVTotDelivered(result,options.outputfile,options.verbose)
            
    if options.action == 'overview':
       session.transaction().start(True)
       result=lumiCalcAPI.lumiForRange(session.nominalSchema(),irunlsdict,beamstatus=pbeammode,norm=normfactor)
       session.transaction().commit()
       if not options.outputfile:
           lumiReport.toScreenOverview(result,options.verbose)
       else:
           lumiReport.toCSVOverview(result,options.outputfile,options.outputfile)
    if options.action == 'lumibyls':
       session.transaction().start(True)
       result=lumiCalcAPI.lumiForRange(session.nominalSchema(),irunlsdict,beamstatus=pbeammode,norm=normfactor)
       session.transaction().commit()
       if not options.outputfile:
           lumiReport.toScreenLumiByLS(result,options.verbose)
       else:
           lumiReport.toCSVLumiByLS(result,options.outputfile,options.verbose)
    if options.action == 'recorded':
        pass
    del session
    del svc 
