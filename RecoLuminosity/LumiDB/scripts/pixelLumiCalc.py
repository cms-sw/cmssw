#!/usr/bin/env python
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
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Lumi Calculation Based on Pixel",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    allowedActions = ['overview', 'delivered', 'recorded', 'lumibyls','checkforupdate']
    #amodetagChoices = [ "PROTPHYS","IONPHYS",'PAPHYS' ]
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
    parser.add_argument('-f','--fill',dest='fillnum',action='store',
                        default=None,required=False,
                        help='fill number (optional) ')
    
    parser.add_argument('--begin',dest='begin',action='store',
                        required=False,
                        type=RegexValidator("^\d\d/\d\d/\d\d \d\d:\d\d:\d\d$","must be form mm/dd/yy hh:mm:ss"),
                        help='min run start time, mm/dd/yy hh:mm:ss')
    
    parser.add_argument('--end',dest='end',action='store',
                        required=False,
                        type=RegexValidator("^\d\d/\d\d/\d\d \d\d:\d\d:\d\d$","must be form mm/dd/yy hh:mm:ss"),
                        help='max run start time, mm/dd/yy hh:mm:ss')    
    #
    #optional args for data and normalization version control
    #
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
    parser.add_argument('--without-correction',
                        dest='withoutCorrection',
                        action='store_true',
                        help='without afterglow correction'
                        )
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
        c=checkforupdate.checkforupdate('pixeltagstatus.txt')
        workingversion=c.runningVersion(cmsswWorkingBase,'pixelLumiCalc.py')
        c.checkforupdate(workingversion)
        exit(0)
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath
        
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
        runlist=lumiCalcAPI.runList(schema,options.fillnum,runmin=None,runmax=None,startT=options.begin,stopT=options.end,l1keyPattern=None,hltkeyPattern=None,amodetag=None,nominalEnergy=None,energyFlut=None,requiretrg=reqTrg,requirehlt=reqHlt,lumitype='PIXEL')
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
    if not options.withoutCorrection:
        session.transaction().start(True)
        finecorrections=lumiCorrections.pixelcorrectionsForRange(session.nominalSchema(),irunlsdict.keys())
        session.transaction().commit()
    if options.verbose:
            print 'afterglow ',finecorrections
    if options.action == 'delivered':
        session.transaction().start(True)
        result=lumiCalcAPI.deliveredLumiForRange(session.nominalSchema(),irunlsdict,amodetag=None,egev=None,beamstatus=None,norm=1.0,finecorrections=finecorrections,driftcorrections=None,usecorrectionv2=False,lumitype='PIXEL',branchName='DATA')
        session.transaction().commit()
        if not options.outputfile:
            lumiReport.toScreenTotDelivered(result,iresults,options.scalefactor,options.verbose)
        else:
            lumiReport.toCSVTotDelivered(result,options.outputfile,iresults,options.scalefactor,options.verbose)           
    if options.action == 'overview':
       session.transaction().start(True)
       result=lumiCalcAPI.lumiForRange(session.nominalSchema(),irunlsdict,amodetag=None,egev=None,beamstatus=None,norm=1.0,finecorrections=finecorrections,driftcorrections=None,usecorrectionv2=False,lumitype='PIXEL',branchName='DATA')
       session.transaction().commit()
       if not options.outputfile:
           lumiReport.toScreenOverview(result,iresults,options.scalefactor,options.verbose)
       else:
           lumiReport.toCSVOverview(result,options.outputfile,iresults,options.scalefactor,options.verbose)
    if options.action == 'lumibyls':
       if not options.hltpath:
           session.transaction().start(True)
           result=lumiCalcAPI.lumiForRange(session.nominalSchema(),irunlsdict,amodetag=None,egev=None,beamstatus=None,norm=1.0,finecorrections=finecorrections,driftcorrections=None,usecorrectionv2=False,lumitype='PIXEL',branchName='DATA')
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
           result=lumiCalcAPI.effectiveLumiForRange(session.nominalSchema(),irunlsdict,hltpathname=hltname,hltpathpattern=hltpat,amodetag=None,egev=None,beamstatus=None,norm=1.0,finecorrections=finecorrections,driftcorrections=None,usecorrectionv2=False,lumitype='PIXEL',branchName='DATA')
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
       result=lumiCalcAPI.effectiveLumiForRange(session.nominalSchema(),irunlsdict,hltpathname=hltname,hltpathpattern=hltpat,amodetag=None,egev=None,beamstatus=None,norm=1.0,finecorrections=finecorrections,driftcorrections=None,usecorrectionv2=False,lumitype='PIXEL',branchName='DATA')
       session.transaction().commit()
       if not options.outputfile:
           lumiReport.toScreenTotEffective(result,iresults,options.scalefactor,options.verbose)
       else:
           lumiReport.toCSVTotEffective(result,options.outputfile,iresults,options.scalefactor,options.verbose)
    del session
    del svc 
