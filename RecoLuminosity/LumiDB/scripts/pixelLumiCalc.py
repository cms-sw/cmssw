#!/usr/bin/env python

########################################################################
# Command to calculate luminosity from HF measurement stored in lumiDB #
#                                                                      #
# Author:      Zhen Xie                                                #
########################################################################

import os,sys,time
import coral

from RecoLuminosity.LumiDB import sessionManager,lumiTime,inputFilesetParser,csvSelectionParser,selectionParser,csvReporter,argparse,CommonUtil,lumiCalcAPI,revisionDML,normDML,lumiReport,lumiCorrections,RegexValidator

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
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Lumi Calculation Based on Pixel",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    allowedActions = ['overview', 'recorded', 'lumibyls']
    #
    # parse arguments
    #  

    # basic arguments
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
                        help='lumi range selection file (optional)')
    #################################################
    #arg to select exact hltpath or pattern
    #################################################
    parser.add_argument('--hltpath',dest='hltpath',action='store',
                        default=None,required=False,
                        help='specific hltpath or hltpath pattern to calculate the effectived luminosity (optional)')
    #################################################
    #versions control
    #################################################
    parser.add_argument('--normtag',dest='normtag',action='store',
                        required=False,
                        help='version of lumi norm/correction')
    parser.add_argument('--datatag',dest='datatag',action='store',
                        required=False,
                        help='version of lumi/trg/hlt data')
    ###############################################
    # run filters
    ###############################################
    parser.add_argument('-f','--fill',dest='fillnum',action='store',
                        default=None,required=False,
                        help='fill number (optional) ')
    
    parser.add_argument('--begin',dest='begin',action='store',
                        default=None,
                        required=False,
                        type=RegexValidator.RegexValidator("^\d\d/\d\d/\d\d \d\d:\d\d:\d\d$|^\d{6}$|^\d{4}$","wrong format"),
                        help='min run start time (mm/dd/yy hh:mm:ss),min fill or min run'
                        ) 
    parser.add_argument('--end',dest='end',action='store',
                        required=False,
                        type=RegexValidator.RegexValidator("^\d\d/\d\d/\d\d \d\d:\d\d:\d\d$|^\d{6}$|^\d{4}$","wrong format"),
                        help='max run start time (mm/dd/yy hh:mm:ss),max fill or max run'
                        )
    parser.add_argument('--minBiasXsec',dest='minbiasxsec',action='store',
                        default=71300.0,
                        type=float,
                        required=False,
                        help='minbias cross-section in ub'
                        )
    #############################################
    #global scale factor
    #############################################       
    parser.add_argument('-n',dest='scalefactor',action='store',
                        type=float,
                        default=1.0,
                        required=False,
                        help='user defined global scaling factor on displayed lumi values,optional')
    #################################################
    #command configuration 
    #################################################
    parser.add_argument('--siteconfpath',dest='siteconfpath',action='store',
                        default=None,
                        required=False,
                        help='specific path to site-local-config.xml file, optional. If path undefined, fallback to cern proxy&server')
    #################################################
    #switches
    #################################################
    parser.add_argument('--without-correction',
                        dest='withoutNorm',
                        action='store_true',
                        help='without afterglow correction'
                        )
    parser.add_argument('--without-checkforupdate',
                        dest='withoutCheckforupdate',
                        action='store_true',
                        help='without check for update'
                        )         
    #parser.add_argument('--verbose',dest='verbose',
    #                    action='store_true',
    #                    help='verbose mode for printing' )
    parser.add_argument('--nowarning',
                        dest='nowarning',
                        action='store_true',
                        help='suppress bad for lumi warnings' )
    parser.add_argument('--debug',dest='debug',
                        action='store_true',
                        help='debug')
    
    options=parser.parse_args()
    if not options.runnumber and not options.inputfile and not options.fillnum and not options.begin:
        raise RuntimeError('at least one run selection argument in [-r,-f,-i,--begin] is required')
    #
    # check working environment
    #
    reqrunmin=None
    reqfillmin=None
    reqtimemin=None
    reqrunmax=None
    reqfillmax=None
    reqtimemax=None
    timeFilter=[None,None]
    
    workingversion='UNKNOWN'
    updateversion='NONE'
    thiscmmd=sys.argv[0]
    if not options.withoutCheckforupdate:
        from RecoLuminosity.LumiDB import checkforupdate
        cmsswWorkingBase=os.environ['CMSSW_BASE']
        if not cmsswWorkingBase:
            print 'Please check out RecoLuminosity/LumiDB from CVS,scram b,cmsenv'
            sys.exit(0)
        c=checkforupdate.checkforupdate('pixeltagstatus.txt')
        workingversion=c.runningVersion(cmsswWorkingBase,'pixelLumiCalc.py',isverbose=False)
        if workingversion:
            updateversionList=c.checkforupdate(workingversion,isverbose=False)
            if updateversionList:
                updateversion=updateversionList[-1][0]
    #
    # check DB environment
    #   
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath
    #############################################################
    #pre-check option compatibility
    #############################################################

    if options.action=='recorded':
        if not options.hltpath:
            raise RuntimeError('argument --hltpath pathname is required for recorded action')
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
    iresults=[]

    session.transaction().start(True)
    if options.runnumber: # if runnumber specified, do not go through other run selection criteria
        irunlsdict[options.runnumber]=None
        rruns=irunlsdict.keys()
    else:  
        runlist=lumiCalcAPI.runList(session.nominalSchema(),options.fillnum,runmin=reqrunmin,runmax=reqrunmax,fillmin=reqfillmin,fillmax=reqfillmax,startT=reqtimemin,stopT=reqtimemax,l1keyPattern=None,hltkeyPattern=None,amodetag=None,nominalEnergy=None,energyFlut=None,requiretrg=False,requirehlt=False,lumitype='PIXEL')
        if options.inputfile:
            (irunlsdict,iresults)=parseInputFiles(options.inputfile)
            rruns=[val for val in runlist if val in irunlsdict.keys()]
            for selectedrun in irunlsdict.keys():#if there's further filter on the runlist,clean input dict
                if selectedrun not in rruns:
                    del irunlsdict[selectedrun]
        else:
            for run in runlist:
                irunlsdict[run]=None
            rruns=irunlsdict.keys()
    
    GrunsummaryData=lumiCalcAPI.runsummaryMap(session.nominalSchema(),irunlsdict)
    ###############################################################
    # check datatag
    ###############################################################
    datatagname=options.datatag
    if not datatagname:
        (datatagid,datatagname)=revisionDML.currentDataTag(session.nominalSchema(),lumitype='PIXEL')
        dataidmap=revisionDML.dataIdsByTagId(session.nominalSchema(),datatagid,runlist=rruns,lumitype='PIXEL',withcomment=False)
        #{run:(lumidataid,trgdataid,hltdataid,())}
    else:
        dataidmap=revisionDML.dataIdsByTagName(session.nominalSchema(),datatagname,runlist=rruns,lumitype='PIXEL',withcomment=False)
        #{run:(lumidataid,trgdataid,hltdataid,())}
    #
    # check normtag and get norm values if required
    #
    normname='NONE'
    normid=0
    normvalueDict={}
    if not options.withoutNorm:
        normname=options.normtag
        if not normname:
            normmap=normDML.normIdByType(session.nominalSchema(),lumitype='PIXEL',defaultonly=True)
            if len(normmap):
                normname=normmap.keys()[0]
                normid=normmap[normname]
        else:
            normid=normDML.normIdByName(session.nominalSchema(),normname)
        if not normid:
            raise RuntimeError('[ERROR] cannot resolve norm/correction')
            sys.exit(-1)
        normvalueDict=normDML.normValueById(session.nominalSchema(),normid) #{since:[corrector(0),{paramname:paramvalue}(1),amodetag(2),egev(3),comment(4)]}
    lumiReport.toScreenHeader(thiscmmd,datatagname,normname,workingversion,updateversion,'PIXEL')
    session.transaction().commit()
    if not dataidmap:
        print '[INFO] No qualified data found, do nothing'
        sys.exit(0)
    
    session.transaction().start(True)
    if options.action == 'overview':
       result=lumiCalcAPI.lumiForIds(session.nominalSchema(),irunlsdict,dataidmap,runsummaryMap=GrunsummaryData,beamstatusfilter=None,timeFilter=timeFilter,normmap=normvalueDict,lumitype='PIXEL')
       lumiReport.toScreenOverview(result,iresults,options.scalefactor,irunlsdict=irunlsdict,noWarning=options.nowarning,toFile=options.outputfile)
    if options.action == 'lumibyls':
       if not options.hltpath:
           result=lumiCalcAPI.lumiForIds(session.nominalSchema(),irunlsdict,dataidmap,runsummaryMap=GrunsummaryData,beamstatusfilter=None,timeFilter=timeFilter,normmap=normvalueDict,lumitype='PIXEL',minbiasXsec=options.minbiasxsec)
           lumiReport.toScreenLumiByLS(result,iresults,options.scalefactor,irunlsdict=irunlsdict,noWarning=options.nowarning,toFile=options.outputfile)
       else:
           hltname=options.hltpath
           hltpat=None
           if hltname=='*' or hltname=='all':
               hltname=None
           elif 1 in [c in hltname for c in '*?[]']: #is a fnmatch pattern
              hltpat=hltname
              hltname=None
           result=lumiCalcAPI.effectiveLumiForIds(session.nominalSchema(),irunlsdict,dataidmap,runsummaryMap=GrunsummaryData,beamstatusfilter=None,timeFilter=timeFilter,normmap=normvalueDict,hltpathname=hltname,hltpathpattern=hltpat,withBXInfo=False,bxAlgo=None,withBeamIntensity=False,lumitype='PIXEL')
           lumiReport.toScreenLSEffective(result,iresults,options.scalefactor,irunlsdict=irunlsdict,noWarning=options.nowarning,toFile=options.outputfile,)
    if options.action == 'recorded':#recorded actually means effective because it needs to show all the hltpaths...
       hltname=options.hltpath
       hltpat=None
       if hltname is not None:
          if hltname=='*' or hltname=='all':
              hltname=None
          elif 1 in [c in hltname for c in '*?[]']: #is a fnmatch pattern
              hltpat=hltname
              hltname=None
       result=lumiCalcAPI.effectiveLumiForIds(session.nominalSchema(),irunlsdict,dataidmap,runsummaryMap=GrunsummaryData,beamstatusfilter=None,normmap=normvalueDict,hltpathname=hltname,hltpathpattern=hltpat,withBXInfo=False,bxAlgo=None,withBeamIntensity=False,lumitype='PIXEL')
       lumiReport.toScreenTotEffective(result,iresults,options.scalefactor,irunlsdict=irunlsdict,noWarning=options.nowarning,toFile=options.outputfile)
    session.transaction().commit()
    del session
    del svc 
