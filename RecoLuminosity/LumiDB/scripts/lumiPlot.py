#!/usr/bin/env python
VERSION='1.00'
import os,sys,datetime,time,commands,csv
from RecoLuminosity.LumiDB import lumiTime,argparse,CommonUtil,matplotRender,sessionManager,lumiCalcAPI,lumiCorrections,dataDML,lumiParameters
from matplotlib.figure import Figure
def parseInputFiles(inputfilename,dbrunlist,optaction):
    '''
    input : inputfile name
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

if __name__=='__main__':
    allowedscales=['linear','log','both']
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     description="Plot integrated luminosity as function of the time variable of choice",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c',
                        dest='connect',
                        action='store',
                        help='connect string to lumiDB',
                        default='frontier://LumiCalc/CMS_LUMI_PROD')
    parser.add_argument('-P',
                        dest='authpath',
                        action='store',
                        help='path to authentication file')
    parser.add_argument('--norm',
                        dest='norm',
                        action='store',
                        default='pp7TeV',
                        help='norm factor name or value')
    parser.add_argument('-i',
                        dest='inputfile',
                        action='store',
                        help='ls range selection file')
    parser.add_argument('-r',
                        dest='runnumber',
                        action='store',
                        type=int,
                        help='run number')
    #parser.add_argument('-f',dest='fillnum',action='store',help='specific fill',default=None)
    parser.add_argument('-b',
                        dest='beamstatus',
                        action='store',
                        help='selection criteria beam status')
    parser.add_argument('--amodetag',
                        dest='amodetag',
                        action='store',
                        help='accelerator mode')
    parser.add_argument('--beamenergy',
                        dest='beamenergy',
                        action='store',
                        type=float,
                        default=3500,
                        help='beamenergy (in GeV) selection criteria')
    parser.add_argument('--beamfluctuation',
                        dest='beamfluctuation',
                        action='store',
                        type=float,
                        default=0.2,
                        help='allowed fraction of beamenergy to fluctuate')
    parser.add_argument('--hltpath',
                        dest='hltpath',
                        action='store',
                        help='specific hltpath to calculate the recorded luminosity. If specified aoverlays the recorded luminosity for the hltpath on the plot')
    parser.add_argument('--beginTime',
                        dest='begintime',
                        action='store',
                        default='03/01/10 00:00:00',
                        help='min run start time,mm/dd/yy hh:mm:ss')
    parser.add_argument('--endTime',
                        dest='endtime',
                        action='store',
                        help='max run start time,mm/dd/yy hh:mm:ss')
    parser.add_argument('--siteconfpath',
                        dest='siteconfpath',
                        action='store',
                        help='specific path to site-local-config.xml file, default to $CMS_PATH/SITECONF/local/JobConfig, if path undefined, fallback to cern proxy&server')
    
    ####plot options
    parser.add_argument('--inplot',
                        dest='inplot',
                        action='store',
                        help='existing base plot(s) in text format')
    parser.add_argument('--outplot',
                        dest='outplot',
                        action='store',
                        help='output plot. By default, a text dump of the plot is produced.')
    parser.add_argument('--yscale',
                        dest='yscale',
                        action='store',
                        default='linear',
                        help='y_scale')
    ####switches
    parser.add_argument('--with-annotation',
                        dest='withannotation',
                        action='store_true',
                        help='annotate boundary run numbers')
    parser.add_argument('--interactive',
                        dest='interactive',
                        action='store_true',
                        help='graphical mode to draw plot in a TK pannel.')
    parser.add_argument('--without-correction',
                        dest='withoutCorrection',
                        action='store_true',
                        help='without fine correction')
    parser.add_argument('--correctionv2',
                        dest='correctionv2',
                        action='store_true',
                        help='apply correction v2')
    parser.add_argument('--correctionv3',
                        dest='correctionv3',
                        action='store_true',
                        help='apply correction v3')
    parser.add_argument('--without-textoutput',
                        dest='withoutTextoutput',
                        action='store_true',
                        help='not to write out text output file')
    parser.add_argument('--verbose',
                        dest='verbose',
                        action='store_true',
                        help='verbose mode, print result also to screen')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    parser.add_argument('action',choices=['run','fill','time','perday','instpeakperday','inst'],help='type of plots')
    options=parser.parse_args()
    actiontofilebasemap={'time':'lumivstime','run':'lumivsrun','fill':'lumivsfill','perday':'lumiperday','instpeakperday':'lumipeak','inst':'inst'}
    outplotfilename = options.outplot
    if not outplotfilename:
        outplotfilename=actiontofilebasemap[options.action]
    outtextfilename = outplotfilename+'.csv'
    if options.withoutTextoutput:
        outtextfilename=None
    lumip=lumiParameters.ParametersObject()
    svc=sessionManager.sessionManager(options.connect,
                                      authpath=options.authpath,
                                      siteconfpath=options.siteconfpath,
                                      debugON=options.debug)
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    lslength=lumip.lslengthsec()
    begtime=options.begintime
    endtime=options.endtime
    lut=lumiTime.lumiTime()
    if not endtime:
        endtime=lut.DatetimeToStr(datetime.datetime.utcnow(),customfm='%m/%d/%y %H:%M:%S')
    pbeammode=None
    normfactor=options.norm
    if options.beamstatus=='stable':
        pbeammode='STABLE BEAMS'
    resultlines=[]
    #
    ##process old plot csv files,if any
    #
    if options.inplot:
        inplot=options.inplot
        inplots=inplot.split('+')
        for ip in inplots:
            f=open(ip,'r')
            reader=csv.reader(f,delimiter=',')
            for row in reader:
                if not row[0].isdigit():continue
                resultlines.append(row)
    #
    ##find runs need to read from DB
    #
    irunlsdict={}
    reqTrg=False
    reqHlt=False
    runsinrange=[]
    session.transaction().start(True)
    schema=session.nominalSchema()
    finecorrections=None
    if options.action=='perday' or options.action=='instpeakperday':
        maxDrawnDay=int(lut.StrToDatetime(begtime,customfm='%m/%d/%y %H:%M:%S').date().toordinal())
        if resultlines:
            for drawnDay in [ int(t[0]) for t in resultlines]:
                if drawnDay>maxDrawnDay:
                    maxDrawnDay=drawnDay
        #print maxDrawnDay
        midnight=datetime.time()
        begT=datetime.datetime.combine(datetime.date.fromordinal(maxDrawnDay),midnight)
        begTStr=lut.DatetimeToStr(begT,customfm='%m/%d/%y %H:%M:%S')
        runsinrange=lumiCalcAPI.runList(schema,runmin=None,runmax=None,startT=begTStr,stopT=endtime,l1keyPattern=None,hltkeyPattern=None,amodetag=options.amodetag,nominalEnergy=options.beamenergy,energyFlut=options.beamfluctuation,requiretrg=reqTrg,requirehlt=reqHlt)
        #print 'runs after maxDrawnDay ',runsinrange
    if options.action=='run' or options.action=='time':
        lastDrawnRun=None
        if resultlines:
            lastDrawnRun=max([int(t[0]) for t in resultlines])
        runsinrange=lumiCalcAPI.runList(schema,runmin=lastDrawnRun,runmax=None,startT=begtime,stopT=endtime,l1keyPattern=None,hltkeyPattern=None,amodetag=options.amodetag,nominalEnergy=options.beamenergy,energyFlut=options.beamfluctuation,requiretrg=reqTrg,requirehlt=reqHlt)
    if options.action=='fill':
        lastDrawnRun=None
        if resultlines:
            lastDrawnFill=max([int(t[0]) for t in resultlines])
        #print lastDrawnFill
        startrun=min([int(t[1]) for t in resultlines if int(t[0])==lastDrawnFill])
        runsinrange=lumiCalcAPI.runList(schema,runmin=startrun,runmax=None,startT=begtime,stopT=endtime,l1keyPattern=None,hltkeyPattern=None,amodetag=options.amodetag,nominalEnergy=options.beamenergy,energyFlut=options.beamfluctuation,requiretrg=reqTrg,requirehlt=reqHlt)
    if options.action=='inst':
        runsinrange.append(options.runnumber)
    session.transaction().commit()
    
    if options.inputfile:
        (irunlsdict,iresults)=parseInputFiles(options.inputfile,runsinrange,options.action)
    else:
        for run in runsinrange:
            irunlsdict[run]=None            
    runlist=irunlsdict.keys()
    runlist.sort()
    if options.verbose:
        print 'runs needed from db ',runlist
    filllist=[]
    #print irunlsdict            
    fig=Figure(figsize=(7.2,5.4),dpi=120)
    m=matplotRender.matplotRender(fig)
    logfig=Figure(figsize=(7.5,5.7),dpi=135)
    mlog=matplotRender.matplotRender(logfig)

    if len(irunlsdict)==0:
        if len(resultlines)!=0:
            print '[INFO] drawing all from old plots'
        else:
            print '[INFO] no old nor new data, do nothing'
            exit(0)

    finecorrections=None
    driftcorrections=None
    if not options.withoutCorrection:
        rruns=irunlsdict.keys()
        session.transaction().start(True)
        schema=session.nominalSchema()
        if options.correctionv2:
            cterms=lumiCorrections.nonlinearV2()
            finecorrections=lumiCorrections.correctionsForRangeV2(schema,rruns,cterms)#constant+nonlinear corrections
            driftcorrections=lumiCorrections.driftcorrectionsForRange(schema,rruns,cterms)
        elif options.correctionv3:
            cterms=lumiCorrections.nonlinearV3()
            finecorrections=lumiCorrections.correctionsForRangeV2(schema,rruns,cterms)#constant+nonlinear corrections
            driftcorrections=lumiCorrections.driftcorrectionsForRange(schema,rruns,cterms)            
        else:#default
            cterms=lumiCorrections.nonlinearSingle()
            finecorrections=lumiCorrections.correctionsForRange(schema,rruns,cterms)
            driftcorrections=None
        session.transaction().commit()
    session.transaction().start(True)    
    lumibyls=lumiCalcAPI.lumiForRange(session.nominalSchema(),irunlsdict,amodetag=options.amodetag,egev=options.beamenergy,beamstatus=pbeammode,norm=normfactor,finecorrections=finecorrections)
    session.transaction().commit()
    fillrunMap={}
    if options.action=='fill' or options.action=='inst':
        fillrunMap=lumiCalcAPI.fillrunMap(schema,runmin=min(irunlsdict.keys()),runmax=max(irunlsdict.keys()))
    session.transaction().commit()
    rawdata={}
    if options.action=='run':
        for run in sorted(lumibyls):
            rundata=lumibyls[run]            
            rawdata.setdefault('Delivered',[]).append((run,sum([t[5] for t in rundata])))
            rawdata.setdefault('Recorded',[]).append((run,sum([t[6] for t in rundata])))
        m.plotSumX_Run(rawdata,resultlines,textoutput=outtextfilename)
    if options.action=='fill':
        for fill in sorted(fillrunMap):
            for run in fillrunMap[fill]:
                rundata=lumibyls[run]
                rawdata.setdefault('Delivered',[]).append((fill,run,sum([t[5] for t in rundata])))
                rawdata.setdefault('Recorded',[]).append((fill,run,sum([t[6] for t in rundata])))
        m.plotSumX_Fill(rawdata,resultlines,textoutput=outtextfilename)
    if options.action=='time':
        for run in sorted(lumibyls):
            rundata=lumibyls[run]
            if len(rundata)!=0:
                rawdata.setdefault('Delivered',[]).append((run,rundata[0][2],rundata[-1][2],sum([t[5] for t in rundata])))
                rawdata.setdefault('Recorded',[]).append((run,rundata[0][2],rundata[-1][2],sum([t[6] for t in rundata])))
        m.plotSumX_Time(rawdata,resultlines,minTime=begtime,maxTime=endtime,textoutput=outtextfilename)
    if options.action=='perday':
        daydict={}
        for run in sorted(lumibyls):
            rundata=lumibyls[run]
            for lsdata in rundata:
                lumilsnum=lsdata[0]
                lsTS=lsdata[2]
                dellum=lsdata[5]
                reclum=lsdata[6]
                daynumber=lsTS.date().toordinal()
                daydict.setdefault(daynumber,[]).append((run,lumilsnum,dellum,reclum))
        for day in sorted(daydict):
            daydata=daydict[day]
            daybeg=str(daydata[0][0])+':'+str(daydata[0][1])
            dayend=str(daydata[-1][0])+':'+str(daydata[-1][1])
            daydel=sum([t[2] for t in daydata])
            dayrec=sum([t[3] for t in daydata])
            rawdata.setdefault('Delivered',[]).append((day,daybeg,dayend,daydel))
            rawdata.setdefault('Recorded',[]).append((day,daybeg,dayend,dayrec))
        #print 'rawdata ',rawdata
        m.plotPerdayX_Time(rawdata,resultlines,minTime=begtime,maxTime=endtime,textoutput=outtextfilename)
    if options.action=='instpeakperday':
        daydict={}#{daynumber:[(runnumber,lumilsnum,inst),..]}
        for run in sorted(lumibyls):
            rundata=lumibyls[run]
            for lsdata in rundata:
                lumilsnum=lsdata[0]
                lsTS=lsdata[2]
                instlum=lsdata[5]/lslength
                daynumber=lsTS.date().toordinal()
                daydict.setdefault(daynumber,[]).append((run,lumilsnum,instlum))
        for day in sorted(daydict):
            daydata=daydict[day]
            daymax_val=0.0
            daymax_run=0
            daymax_ls=0
            for datatp in daydata:
                if datatp[2]>daymax_val:
                    daymax_val=datatp[2]
                    daymax_run=datatp[0]
                    daymax_ls=datatp[1]
            rawdata.setdefault('Delivered',[]).append((day,daymax_run,daymax_ls,daymax_val))
        m.plotPeakPerday_Time(rawdata,resultlines,minTime=begtime,maxTime=endtime,textoutput=outtextfilename)
    if options.action=='inst':
        thisfillnumber=fillrunMap.keys()[0]
        starttime=0
        stoptime=0
        rawxdata=[]
        rawydata={}
        for run,rundata in lumibyls.items():
            rundata.sort()
            totlumils=0
            totcmsls=0
            starttime=rundata[0][2]
            stoptime=rundata[-1][2]
            for lsdata in rundata:
                lumilsnum=lsdata[0]
                totlumils+=1
                cmslsnum=lsdata[1]
                if cmslsnum!=0:
                    totcmsls+=1
                lsTS=lsdata[2]
                dellumi=lsdata[5]
                reclumi=lsdata[6]
                rawydata.setdefault('Delivered',[]).append(dellumi)
                rawydata.setdefault('Recorded',[]).append(reclumi)
            rawxdata=[run,thisfillnumber,starttime,stoptime,totlumils,totcmsls]
        m.plotInst_RunLS(rawxdata,rawydata,textoutput=None)

    if options.interactive:
        if options.yscale=='linear':
            m.drawInteractive()
            exit(0)
        if options.yscale=='log':
            mlog.drawInteractive()
            exit(0)
    if options.yscale=='linear':
        m.drawPNG(outplotfilename+'.png')
    elif options.yscale=='log':
        mlog.drawPNG(outplotfilename+'_log.png')
    else:
        m.drawPNG(outplotfilename+'.png')            
        mlog.drawPNG(outplotfilename+'_log.png')
