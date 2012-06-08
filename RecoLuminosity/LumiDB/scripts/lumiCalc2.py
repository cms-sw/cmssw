#!/usr/bin/env python
VERSION='2.00'
import os,sys,time,re
import coral
from RecoLuminosity.LumiDB import sessionManager,lumiTime,inputFilesetParser,csvSelectionParser,selectionParser,csvReporter,argparse,CommonUtil,lumiCalcAPI,lumiReport,lumiCorrections

class RegexValidator(object):
    def __init__(self, pattern, statement=None):
        self.pattern = re.compile(pattern)
        self.statement = statement
        if not self.statement:
            self.statement = "must match pattern %s" % self.pattern

    def __call__(self, string):
        match = self.pattern.search(string)
        if not match:
            raise ValueError(self.statement)
        return string 
        
beamChoices=['PROTPHYS','IONPHYS','PAPHYS']

def parseInputFiles(inputfilename,dbrunlist,optaction):
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
        if runinfile not in dbrunlist:
            continue
        if optaction=='delivered':#for delivered we care only about selected runs
            selectedrunlsInDB[runinfile]=None
        else:
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
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Lumi Calculation",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    allowedActions = ['overview', 'delivered', 'recorded', 'lumibyls','lumibylsXing','status','checkforupdate']
    beamModeChoices = [ "stable", "quiet", "either"]
    amodetagChoices = [ "PROTPHYS","IONPHYS",'PAPHYS' ]
    xingAlgoChoices =[ "OCC1","OCC2","ET"]
    #
    # parse arguments
    #  
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
    #
    #optional arg to select exact run and ls
    #
    parser.add_argument('-i',dest='inputfile',action='store',
                        required=False,
                        help='lumi range selection file (optional)')
    #
    #optional arg to select exact hltpath or pattern
    #
    parser.add_argument('--hltpath',dest='hltpath',action='store',
                        default=None,required=False,
                        help='specific hltpath or hltpath pattern to calculate the effectived luminosity (optional)')
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
                        help='specific accelerator mode choices [PROTOPHYS,IONPHYS,PAPHYS] (optional)')
    parser.add_argument('--beamenergy',dest='beamenergy',action='store',
                        type=float,
                        default=None,
                        help='nominal beam energy in GeV')
    parser.add_argument('--beamfluctuation',dest='beamfluctuation',
                        type=float,action='store',
                        default=0.2,
                        required=False,
                        help='fluctuation in fraction allowed to nominal beam energy, default 0.2, to be used together with -beamenergy  (optional)'
                        )
    parser.add_argument('--begin',dest='begin',action='store',
                        default=None,
                        required=False,
                        type=RegexValidator("^\d\d/\d\d/\d\d \d\d:\d\d:\d\d$","must be form mm/dd/yy hh:mm:ss"),
                        help='min run start time, mm/dd/yy hh:mm:ss (optional)'
                        )
    parser.add_argument('--end',dest='end',action='store',
                        default=None,
                        required=False,
                        type=RegexValidator("^\d\d/\d\d/\d\d \d\d:\d\d:\d\d$","must be form mm/dd/yy hh:mm:ss"),
                        help='max run start time, mm/dd/yy hh:mm:ss (optional)'
                        )    
    #
    #optional args to filter ls
    #
    parser.add_argument('--xingMinLum', dest = 'xingMinLum',
                        type=float,
                        default=1e-03,
                        required=False,
                        help='Minimum luminosity considered for lumibylsXing action, default=1e-03')
    parser.add_argument('--xingAlgo', dest = 'xingAlgo',
                        default='OCC1',
                        required=False,
                        help='algorithm name for per-bunch lumi ')
    #
    #optional args for data and normalization version control
    #
    parser.add_argument('--lumiversion',dest='lumiversion',action='store',
                        default=None,
                        required=False,
                        help='data version, optional')
    parser.add_argument('--norm',dest='normfactor',action='store',
                        default=None,
                        required=False,
                        help='use specify the name or the value of the normalization to use,optional')
    parser.add_argument('-n',dest='scalefactor',action='store',
                        type=float,
                        default=1.0,
                        required=False,
                        help='user defined global scaling factor on displayed lumi values,optional')
    #
    #command configuration 
    #
    parser.add_argument('--siteconfpath',dest='siteconfpath',action='store',
                        default=None,
                        required=False,
                        help='specific path to site-local-config.xml file, optional. If path undefined, fallback to cern proxy&server')
    #
    #switches
    #
    parser.add_argument('--without-correction',dest='withoutFineCorrection',action='store_true',
                        help='without fine correction on calibration' )
    parser.add_argument('--correctionv3',dest='correctionv3',action='store_true',
                        help='apply correction v3' )
    parser.add_argument('--verbose',dest='verbose',action='store_true',
                        help='verbose mode for printing' )
    parser.add_argument('--nowarning',dest='nowarning',action='store_true',
                        help='suppress bad for lumi warnings' )
    parser.add_argument('--debug',dest='debug',action='store_true',
                        help='debug')
    
    options=parser.parse_args()
    if options.action=='checkforupdate':
        from RecoLuminosity.LumiDB import checkforupdate
        cmsswWorkingBase=os.environ['CMSSW_BASE']
        c=checkforupdate.checkforupdate()
        workingversion=c.runningVersion(cmsswWorkingBase,'lumiCalc2.py')
        c.checkforupdate(workingversion)
        exit(0)
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath
        
    pbeammode = None
    normfactor=options.normfactor
    if options.beammode=='stable':
        pbeammode    = 'STABLE BEAMS'
    if options.verbose:
        print 'General configuration'
        print '\tconnect: ',options.connect
        print '\tauthpath: ',options.authpath
        print '\tlumi data version: ',options.lumiversion
        print '\tsiteconfpath: ',options.siteconfpath
        print '\toutputfile: ',options.outputfile
        print '\tscalefactor: ',options.scalefactor        
        if options.action=='recorded' and options.hltpath:
            print 'Action: effective luminosity in hltpath: ',options.hltpath
        else:
            print 'Action: ',options.action
        if options.normfactor:
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
        
    svc=sessionManager.sessionManager(options.connect,
                                      authpath=options.authpath,
                                      siteconfpath=options.siteconfpath,
                                      debugON=options.debug)
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    
    irunlsdict={}
    iresults=[]
    if options.runnumber: # if runnumber specified, do not go through other run selection criteria
        irunlsdict[options.runnumber]=None
    else:
        reqTrg=False
        reqHlt=False
        if options.action=='recorded':
            reqTrg=True
            reqHlt=True
        session.transaction().start(True)
        schema=session.nominalSchema()
        runlist=lumiCalcAPI.runList(schema,options.fillnum,runmin=None,runmax=None,startT=options.begin,stopT=options.end,l1keyPattern=None,hltkeyPattern=None,amodetag=options.amodetag,nominalEnergy=options.beamenergy,energyFlut=options.beamfluctuation,requiretrg=reqTrg,requirehlt=reqHlt)
        session.transaction().commit()
        if options.inputfile:
            (irunlsdict,iresults)=parseInputFiles(options.inputfile,runlist,options.action)
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

    finecorrections=None
    driftcorrections=None
    if not options.withoutFineCorrection:
        rruns=irunlsdict.keys()
        schema=session.nominalSchema()
        session.transaction().start(True)
        if options.correctionv3:
            cterms=lumiCorrections.nonlinearV3()                   
        else:#default            
            cterms=lumiCorrections.nonlinearV2()
        finecorrections=lumiCorrections.correctionsForRangeV2(schema,rruns,cterms)#constant+nonlinear corrections
        driftcorrections=lumiCorrections.driftcorrectionsForRange(schema,rruns,cterms)
        if options.verbose:
            print finecorrections,driftcorrections    
        session.transaction().commit()
        
    if options.action == 'delivered':
        session.transaction().start(True)
        #print irunlsdict
        result=lumiCalcAPI.deliveredLumiForRange(session.nominalSchema(),irunlsdict,amodetag=options.amodetag,egev=options.beamenergy,beamstatus=pbeammode,norm=normfactor,finecorrections=finecorrections,driftcorrections=driftcorrections,usecorrectionv2=True)
        session.transaction().commit()
        if not options.outputfile:
            lumiReport.toScreenTotDelivered(result,iresults,options.scalefactor,options.verbose)
        else:
            lumiReport.toCSVTotDelivered(result,options.outputfile,iresults,options.scalefactor,options.verbose)           
    if options.action == 'overview':
       session.transaction().start(True)
       result=lumiCalcAPI.lumiForRange(session.nominalSchema(),irunlsdict,amodetag=options.amodetag,egev=options.beamenergy,beamstatus=pbeammode,norm=normfactor,finecorrections=finecorrections,driftcorrections=driftcorrections,usecorrectionv2=True)
       session.transaction().commit()
       if not options.outputfile:
           lumiReport.toScreenOverview(result,iresults,options.scalefactor,options.verbose)
       else:
           lumiReport.toCSVOverview(result,options.outputfile,iresults,options.scalefactor,options.verbose)
    if options.action == 'lumibyls':
       if not options.hltpath:
           session.transaction().start(True)
           result=lumiCalcAPI.lumiForRange(session.nominalSchema(),irunlsdict,amodetag=options.amodetag,egev=options.beamenergy,beamstatus=pbeammode,norm=normfactor,finecorrections=finecorrections,driftcorrections=driftcorrections,usecorrectionv2=True)
           session.transaction().commit()
           if not options.outputfile:
               lumiReport.toScreenLumiByLS(result,iresults,options.scalefactor,options.verbose)
           else:
               lumiReport.toCSVLumiByLS(result,options.outputfile,iresults,options.scalefactor,options.verbose)
       else:
           hltname=options.hltpath
           hltpat=None
           if hltname=='*' or hltname=='all':
               hltname=None
           elif 1 in [c in hltname for c in '*?[]']: #is a fnmatch pattern
              hltpat=hltname
              hltname=None
           session.transaction().start(True)
           result=lumiCalcAPI.effectiveLumiForRange(session.nominalSchema(),irunlsdict,hltpathname=hltname,hltpathpattern=hltpat,amodetag=options.amodetag,egev=options.beamenergy,beamstatus=pbeammode,norm=normfactor,finecorrections=finecorrections,driftcorrections=driftcorrections,usecorrectionv2=True)
           session.transaction().commit()
           if not options.outputfile:
               lumiReport.toScreenLSEffective(result,iresults,options.scalefactor,options.verbose)
           else:
               lumiReport.toCSVLSEffective(result,options.outputfile,iresults,options.scalefactor,options.verbose)
    if options.action == 'recorded':#recorded actually means effective because it needs to show all the hltpaths...
       session.transaction().start(True)
       hltname=options.hltpath
       hltpat=None
       if hltname is not None:
          if hltname=='*' or hltname=='all':
              hltname=None
          elif 1 in [c in hltname for c in '*?[]']: #is a fnmatch pattern
              hltpat=hltname
              hltname=None
       result=lumiCalcAPI.effectiveLumiForRange(session.nominalSchema(),irunlsdict,hltpathname=hltname,hltpathpattern=hltpat,amodetag=options.amodetag,egev=options.beamenergy,beamstatus=pbeammode,norm=normfactor,finecorrections=finecorrections,driftcorrections=driftcorrections,usecorrectionv2=True)
       session.transaction().commit()
       if not options.outputfile:
           lumiReport.toScreenTotEffective(result,iresults,options.scalefactor,options.verbose)
       else:
           lumiReport.toCSVTotEffective(result,options.outputfile,iresults,options.scalefactor,options.verbose)
    if options.action == 'lumibylsXing':
       session.transaction().start(True)
       result=lumiCalcAPI.lumiForRange(session.nominalSchema(),irunlsdict,amodetag=options.amodetag,egev=options.beamenergy,beamstatus=pbeammode,norm=normfactor,xingMinLum=options.xingMinLum,withBeamIntensity=False,withBXInfo=True,bxAlgo=options.xingAlgo,finecorrections=finecorrections,driftcorrections=True)
       session.transaction().commit()           
       if not options.outputfile:
           lumiReport.toScreenLumiByLS(result,iresults,options.scalefactor,options.verbose)
       else:
           lumiReport.toCSVLumiByLSXing(result,options.scalefactor,options.outputfile)
    del session
    del svc 
