#!/usr/bin/env python

###############################################################################
# Command to display runsummary, L1,HLT and beam data used in lumi caculation #
#                                                                             #
# Author:      Zhen Xie                                                       #
###############################################################################

import os,sys,time
from RecoLuminosity.LumiDB import sessionManager,lumiTime,inputFilesetParser,csvSelectionParser,csvReporter,argparse,CommonUtil,lumiCalcAPI,lumiReport,RegexValidator,lumiTime,revisionDML

def parseInputFiles(inputfilename):
    '''
    output ({run:[cmsls,cmsls,...]},[[resultlines]])
    '''
    selectedrunlsInDB={}
    resultlines=[]
    p=inputFilesetParser.inputFilesetParser(inputfilename)
    runlsbyfile=p.runsandls()
    selectedProcessedRuns=p.selectedRunsWithresult()
    selectedNonProcessedRuns=p.selectedRunsWithoutresult()
    resultlines=p.resultlines()
    for runinfile in selectedNonProcessedRuns:
        selectedrunlsInDB[runinfile]=runlsbyfile[runinfile]
    return (selectedrunlsInDB,resultlines)

##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Additional information needed in the lumi calculation",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    allowedActions = ['hltbyls','hltmenu','trgbyls', 'beambyls','runsummary']
    amodetagChoices = [ "PROTPHYS","IONPHYS","PAPHYS" ]
    beamModeChoices = ["stable"]
    #
    # parse arguments
    #  
    #
    ################################################
    # basic arguments
    ################################################
    #
    parser.add_argument('action',choices=allowedActions,
                        help='command actions')
    parser.add_argument('-c',dest='connect',action='store',
                        required=False,
                        help='connect string to lumiDB,optional',
                        default='frontier://LumiCalc/CMS_LUMI_PROD')
    parser.add_argument('-P',dest='authpath',action='store',
                        required=False,
                        help='path to authentication file (optional)')
    parser.add_argument('-r',dest='runnumber',action='store',
                        type=int,
                        required=False,
                        help='run number (optional)')
    parser.add_argument('-o',dest='outputfile',action='store',
                        required=False,
                        help='output to csv file (optional)')
    #################################################
    #arg to select exact run and ls
    #################################################
    parser.add_argument('-i',dest='inputfile',action='store',
                        required=False,
                        help='run/ls selection file (optional)')
    parser.add_argument('--name',dest='name',action='store',
                       help='hltpath/l1bit name/pattern'
                       )
    #
    #optional args to filter *runs*, they do not select on LS level.
    #
    parser.add_argument('-b',dest='beammode',action='store',
                        choices=beamModeChoices,
                        required=False,
                        help='beam mode choices [stable] (optional)')
    parser.add_argument('-f','--fill',dest='fillnum',action='store',
                        default=None,required=False,
                        help='fill number (optional) ')
    parser.add_argument('--amodetag',dest='amodetag',action='store',
                        choices=amodetagChoices,
                        required=False,
                        help='specific accelerator mode choices [PROTOPHYS,IONPHYS] (optional)')
    parser.add_argument('--beamenergy',dest='beamenergy',action='store',
                        type=float,
                        default=None,
                        help='nominal beam energy in GeV')
    parser.add_argument('--beamfluctuation',dest='beamfluctuation',
                        type=float,action='store',
                        default=0.2,
                        required=False,
                        help='fluctuation in fraction allowed to nominal beam energy, default 0.2, to be used together with -beamenergy  (optional)')
    parser.add_argument('--minintensity',dest='minintensity',
                        type=float,action='store',
                        default=0.1,
                        required=False,
                        help='filter on beam intensity , effective with --with-beamintensity (optional)')
    parser.add_argument('--begin',dest='begin',action='store',
                        default=None,
                        required=False,
                        type=RegexValidator.RegexValidator("^\d\d/\d\d/\d\d \d\d:\d\d:\d\d$","must be form mm/dd/yy hh:mm:ss"),
                        help='min run start time, mm/dd/yy hh:mm:ss)'
                        )
    parser.add_argument('--end',dest='end',action='store',
                        default=None,
                        required=False,
                        type=RegexValidator.RegexValidator("^\d\d/\d\d/\d\d \d\d:\d\d:\d\d$","must be form mm/dd/yy hh:mm:ss"),
                        help='max run start time, mm/dd/yy hh:mm:ss'
                        )
    #################################################
    #versions control
    #################################################
    parser.add_argument('--datatag',dest='datatag',action='store',
                        required=False,
                        help='version of lumi/trg/hlt data'
                        )
    #
    #command configuration 
    #
    parser.add_argument('--siteconfpath',dest='siteconfpath',action='store',
                        default=None,
                        required=False,
                        help='specific path to site-local-config.xml file, optional. If path undefined, fallback to cern proxy&server')
    #################################################
    #switches
    #################################################                    
    parser.add_argument('--with-beamintensity',
                        dest='withbeamintensity',
                        action='store_true',
                        help='dump beam intensity'
                        )
    parser.add_argument('--without-mask',
                        dest='withoutmask',
                        action='store_true',
                        help='not considering trigger mask'
                        )
    parser.add_argument('--verbose',
                        dest='verbose',
                        action='store_true',
                        help='verbose mode for printing' )
    parser.add_argument('--nowarning',
                        dest='nowarning',
                        action='store_true',
                        help='suppress bad for lumi warnings'
                        )
    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        help='debug')
    
    options=parser.parse_args()
    if not options.runnumber and not options.inputfile and not options.fillnum and not options.begin:
        raise RuntimeError('at least one run selection argument in [-r,-f,-i,--begin] is required')
    reqrunmin=None
    reqfillmin=None
    reqtimemin=None
    reqrunmax=None
    reqfillmax=None
    reqtimemax=None
    timeFilter=[None,None]
    pbeammode = None
    iresults=[]
    reqTrg=False
    reqHlt=False
    if options.beammode=='stable':
        pbeammode = 'STABLE BEAMS'
    if options.action=='trgbyls':
        reqTrg=True
    if options.action=='hltbyls':
        reqHlt=True
    if options.runnumber:
        reqrunmax=options.runnumber
        reqrunmin=options.runnumber
    if options.fillnum:
        reqfillmin=options.fillnum
        reqfillmax=options.fillnum
    if options.begin:
        (reqrunmin,reqfillmin,reqtimemin)=CommonUtil.parseTime(options.begin)
        if reqtimemin:
            lute=lumiTime.lumiTime()
            reqtimeminT=lute.StrToDatetime(reqtimemin,customfm='%m/%d/%y %H:%M:%S')
            timeFilter[0]=reqtimeminT
    if options.end:
        (reqrunmax,reqfillmax,reqtimemax)=CommonUtil.parseTime(options.end)
        if reqtimemax:
            lute=lumiTime.lumiTime()
            reqtimemaxT=lute.StrToDatetime(reqtimemax,customfm='%m/%d/%y %H:%M:%S')
            timeFilter[1]=reqtimemaxT
    sname=options.name
    isdetail=False
    spattern=None
    if sname is not None:
        isdetail=True
        if sname=='*' or sname=='all':
            sname=None
        elif 1 in [c in sname for c in '*?[]']: #is a fnmatch pattern
            spattern=sname
            sname=None
    if  options.action == 'beambyls' and options.withbeamintensity and not options.outputfile:
        print '[warning] --with-beamintensity must write data to a file, none specified using default "beamintensity.csv"'
        options.outputfile='beamintensity.csv'
        
    ##############################################################
    # check working environment
    ##############################################################
    #
    # check DB environment
    #    
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath

    svc=sessionManager.sessionManager(options.connect,
                                      authpath=options.authpath,
                                      siteconfpath=options.siteconfpath,
                                      debugON=options.debug)
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    
    ##############################################################
    # check run/ls list
    ##############################################################
    
    irunlsdict={}
    rruns=[]
    session.transaction().start(True)
    filerunlist=None
    if options.inputfile:
        (irunlsdict,iresults)=parseInputFiles(options.inputfile)
        filerunlist=irunlsdict.keys()

    datatagname=options.datatag
    if not datatagname:
        (datatagid,datatagname)=revisionDML.currentDataTag(session.nominalSchema())
    else:
        datatagid=revisionDML.getDataTagId(session.nominalSchema(),datatagname)

    dataidmap=lumiCalcAPI.runList(session.nominalSchema(),datatagid,runmin=reqrunmin,runmax=reqrunmax,fillmin=reqfillmin,fillmax=reqfillmax,startT=reqtimemin,stopT=reqtimemax,l1keyPattern=None,hltkeyPattern=None,amodetag=options.amodetag,nominalEnergy=options.beamenergy,energyFlut=options.beamfluctuation,requiretrg=reqTrg,requirehlt=reqHlt,preselectedruns=filerunlist)
    if not dataidmap:
        print '[INFO] No qualified run found, do nothing'
        sys.exit(14)
    rruns=[]
    #crosscheck dataid value
    for irun,(lid,tid,hid) in dataidmap.items():
        if not lid:
            print '[INFO] No qualified lumi data found for run, ',irun
        if reqTrg and not tid:
            print '[INFO] No qualified trg data found for run ',irun
            continue
        if reqHlt and not hid:
            print '[INFO] No qualified hlt data found for run ',irun
            continue
        rruns.append(irun)
    if not irunlsdict: #no file
        irunlsdict=dict(list(zip(rruns,[None]*len(rruns))))
    else:
        for selectedrun in irunlsdict.keys():#if there's further filter on the runlist,clean input dict
            if selectedrun not in rruns:
                del irunlsdict[selectedrun]
    if not irunlsdict:
        print '[INFO] No qualified run found, do nothing'
        sys.exit(13)
        
    session.transaction().commit()
    thiscmmd=sys.argv[0]
    lumiReport.toScreenHeader(thiscmmd,datatagname,'n/a','n/a','n/a','n/a')
    
    if options.action == 'trgbyls':
        session.transaction().start(True)
        result=lumiCalcAPI.trgForIds(session.nominalSchema(),irunlsdict,dataidmap,trgbitname=sname,trgbitnamepattern=spattern,withL1Count=True,withPrescale=True)
        session.transaction().commit()
        if not options.outputfile:
            lumiReport.toScreenLSTrg(result,iresults=iresults,irunlsdict=irunlsdict,noWarning=options.nowarning,withoutmask=options.withoutmask)
        else:
            lumiReport.toScreenLSTrg(result,iresults=iresults,irunlsdict=irunlsdict,noWarning=options.nowarning,toFile=options.outputfile,withoutmask=options.withoutmask)
    #print result
        sys.exit(0)
    if options.action == 'hltbyls':
        if not options.name:
            print '[ERROR] --name option is required by hltbyls, do nothing'
            sys.exit(0)
        withL1Pass=True
        withHLTAccept=True
        session.transaction().start(True)
        result=lumiCalcAPI.hltForIds(session.nominalSchema(),irunlsdict,dataidmap,hltpathname=sname,hltpathpattern=spattern,withL1Pass=withL1Pass,withHLTAccept=withHLTAccept)
        session.transaction().commit()
        if not options.outputfile:
            lumiReport.toScreenLSHlt(result,iresults=iresults)
        else:
            lumiReport.toScreenLSHlt(result,iresults=iresults,toFile=options.outputfile)
    if options.action == 'hltmenu':
        session.transaction().start(True)
        result=lumiCalcAPI.hltpathsForRange(session.nominalSchema(),irunlsdict,hltpathname=sname,hltpathpattern=spattern)
        session.transaction().commit()
        #print result
        if not options.outputfile:
            lumiReport.toScreenConfHlt(result,iresults)
        else:
            lumiReport.toScreenConfHlt(result,iresults,toFile=options.outputfile)
    if options.action == 'beambyls':
        session.transaction().start(True)
        dumpbeamintensity=False
        if options.outputfile and options.verbose:
            dumpbeamintensity=True
        result=lumiCalcAPI.beamForIds(session.nominalSchema(),irunlsdict,dataidmap,withBeamIntensity=options.withbeamintensity,minIntensity=options.minintensity)
        session.transaction().commit()
        if not options.outputfile:
            lumiReport.toScreenLSBeam(result,iresults=iresults,dumpIntensity=False)
        else:
            lumiReport.toScreenLSBeam(result,iresults=iresults,dumpIntensity=options.withbeamintensity,toFile=options.outputfile)
    if options.action == 'runsummary':
        session.transaction().start(True)
        result=lumiCalcAPI.runsummary(session.nominalSchema(),irunlsdict)
        session.transaction().commit()
        c=lumiTime.lumiTime()
        for r in result:
            run=r[0]
            fill='n/a'
            if r[5]:
                fill=str(r[5])
            starttime=c.StrToDatetime(r[7])
            starttime=starttime.strftime('%m/%d/%y %H:%M:%S')
            stoptime=c.StrToDatetime(r[8])
            stoptime=stoptime.strftime('%m/%d/%y %H:%M:%S')
            l1key=r[1]
            hltkey=r[4]
            amodetag=r[2]
            egev='n/a'
            if r[3]:
                egev=str(r[3])
            sequence=r[6]
            print '==='
            print 'Run ',str(run),' Fill ',fill,' Amodetag ',amodetag,' egev ',egev
            print '\tStart '+starttime,'                  ',' Stop ',stoptime
            print '\tL1key ',l1key,' HLTkey ',hltkey
    del session
    del svc 
