#!/usr/bin/env python
VERSION='1.00'
import os,sys,datetime
import coral
from RecoLuminosity.LumiDB import lumiTime,argparse,nameDealer,selectionParser,hltTrgSeedMapper,connectstrParser,cacheconfigParser,matplotRender,lumiQueryAPI,inputFilesetParser,csvReporter,CommonUtil
from matplotlib.figure import Figure
class constants(object):
    def __init__(self):
        self.NORM=1.0
        self.LUMIVERSION='0001'
        self.NBX=3564
        self.VERBOSE=False
    def defaultfrontierConfigString(self):
        return """<frontier-connect><proxy url="http://cmst0frontier.cern.ch:3128"/><proxy url="http://cmst0frontier.cern.ch:3128"/><proxy url="http://cmst0frontier1.cern.ch:3128"/><proxy url="http://cmst0frontier2.cern.ch:3128"/><server url="http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier1.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier2.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier3.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier4.cern.ch:8000/FrontierInt"/></frontier-connect>"""

def getLumiOrderByLS(dbsession,c,runList,selectionDict,hltpath='',beamstatus=None,beamenergy=None,beamfluctuation=None):
    '''
    input:  runList[runnum], selectionDict{runnum:[ls]}
    output: [[runnumber,runstarttime,lsnum,lsstarttime,delivered,recorded,recordedinpath]]
    '''
    #print 'getLumiOrderByLS selectionDict seen ',selectionDict
    t=lumiTime.lumiTime()
    result=[]#[[runnumber,runstarttime,lsnum,lsstarttime,delivered,recorded]]
    dbsession.transaction().start(True)
    sortedresult=[]
    #print 'runlist ',runList
    for runnum in runList:
        delivered=0.0
        recorded=0.0       
        #print 'looking for run ',runnum
        q=dbsession.nominalSchema().newQuery()
        runsummary=lumiQueryAPI.runsummaryByrun(q,runnum)
        del q
        runstarttimeStr=runsummary[3]
        if len(runstarttimeStr)==0:
            if c.VERBOSE: print 'warning request run ',runnum,' has no runsummary, skip'
            continue
        if len(selectionDict)!=0 and not selectionDict.has_key(runnum):
            if runnum<max(selectionDict.keys()):
                result.append([runnum,runstarttimeStr,1,t.StrToDatetime(runstarttimeStr),0.0,0.0])
            continue
        #print 'runsummary ',runsummary
        lumitrginfo={}
        q=dbsession.nominalSchema().newQuery()
        lumitrginfo=lumiQueryAPI.lumisummarytrgbitzeroByrun(q,runnum,c.LUMIVERSION,beamstatus,beamenergy,beamfluctuation) #q2
        del q
        #print 'lumitrginfo ',lumitrginfo
        if len(lumitrginfo)==0: #if no qualified cross lumi-trg found, try lumionly
            #result.append([runnum,runstarttimeStr,1,t.StrToDatetime(runstarttimeStr),0.0,0.0])
            q=dbsession.nominalSchema().newQuery()
            lumiinfobyrun=lumiQueryAPI.lumisummaryByrun(q,runnum,c.LUMIVERSION,beamstatus,beamenergy,beamfluctuation) #q3
            del q
            if len(lumiinfobyrun)!=0: #if lumionly has qualified data means trg has no data
                print 'warning request run ',runnum,' has no trigger data, calculate delivered only'
                for perlsdata in lumiinfobyrun:
                    cmslsnum=perlsdata[0]
                    instlumi=perlsdata[1]
                    norbit=perlsdata[2]
                    startorbit=perlsdata[3]
                    lsstarttime=t.OrbitToTime(runstarttimeStr,startorbit)
                    lslength=t.bunchspace_s*t.nbx*norbit
                    delivered=instlumi*lslength
                    result.append([runnum,runstarttimeStr,cmslsnum,lsstarttime,delivered,0.0])
            else:
                #print 'run '+str(runnum)+' has no qualified data '
                lsstarttime=t.OrbitToTime(runstarttimeStr,0)
                result.append([runnum,runstarttimeStr,1,lsstarttime,0.0,0.0])
        else:
            norbits=lumitrginfo.values()[0][1]
            lslength=t.bunchspace_s*t.nbx*norbits
            trgbitinfo={}
            for cmslsnum,valuelist in lumitrginfo.items():
                instlumi=valuelist[0]
                startorbit=valuelist[2]
                bitzero=valuelist[5]
                deadcount=valuelist[6]
                prescale=valuelist[-1]
                lsstarttime=t.OrbitToTime(runstarttimeStr,startorbit)        
                if len(selectionDict)!=0 and not (cmslsnum in selectionDict[runnum]):
                   #if there's a selection list but cmslsnum is not selected,set to 0
                   result.append([runnum,runstarttimeStr,cmslsnum,lsstarttime,0.0,0.0])
                   continue
                delivered=instlumi*lslength
                if valuelist[5]==0:#bitzero==0 means no beam,do nothing
                    recorded=0.0
                else:
                    deadfrac=float(deadcount)/float(float(bitzero)*float(prescale))
                    if(deadfrac<1.0):
                        recorded=delivered*(1.0-deadfrac)
                result.append([runnum,runstarttimeStr,cmslsnum,lsstarttime,delivered,recorded])
                #print 'result : ',result
    dbsession.transaction().commit()
    transposedResult=CommonUtil.transposed(result)
    lstimes=transposedResult[3]
    lstimes.sort()
    for idx,lstime in enumerate(lstimes):
        sortedresult.append(result[idx])
    if c.VERBOSE:
        print sortedresult
    return sortedresult           

def getLumiInfoForRuns(dbsession,c,runList,selectionDict,hltpath='',beamstatus=None,beamenergy=None,beamfluctuation=0.0):
    '''
    input: runList[runnum], selectionDict{runnum:[ls]}
    output:{runnumber:[delivered,recorded,recordedinpath] }
    '''
    t=lumiTime.lumiTime()
    result={}#runnumber:[lumisumoverlumils,lumisumovercmsls-deadtimecorrected,lumisumovercmsls-deadtimecorrected*hltcorrection_hltpath]
    #print 'selectionDict seen ',selectionDict
    dbsession.transaction().start(True)
    for runnum in runList:
        totallumi=0.0
        delivered=0.0
        recorded=0.0 
        recordedinpath=0.0
        if len(selectionDict)!=0 and not selectionDict.has_key(runnum):
            if runnum<max(selectionDict.keys()):
                result[runnum]=[0.0,0.0,0.0]
            continue
        #print 'looking for run ',runnum
        q=dbsession.nominalSchema().newQuery()
        totallumi=lumiQueryAPI.lumisumByrun(q,runnum,c.LUMIVERSION,beamstatus,beamenergy,beamfluctuation) #q1
        del q
        if not totallumi:
            result[runnum]=[0.0,0.0,0.0]
            if c.VERBOSE: print 'run ',runnum,' does not exist or has no lumi, skip'
            continue
        lumitrginfo={}
        hltinfo={}
        hlttrgmap={}
        q=dbsession.nominalSchema().newQuery()
        lumitrginfo=lumiQueryAPI.lumisummarytrgbitzeroByrun(q,runnum,c.LUMIVERSION,beamstatus,beamenergy,beamfluctuation) #q2
        del q
        if len(lumitrginfo)==0:
            q=dbsession.nominalSchema().newQuery()
            lumiinfobyrun=lumiQueryAPI.lumisummaryByrun(q,runnum,c.LUMIVERSION,beamstatus,beamenergy,beamfluctuation) #q3
            del q
            if len(lumiinfobyrun)!=0:
                print 'warning request run ',runnum,' has no trigger data, calculate delivered only'
            for perlsdata in lumiinfobyrun:
                cmslsnum=perlsdata[0]
                instlumi=perlsdata[1]
                norbit=perlsdata[2]
                lslength=t.bunchspace_s*t.nbx*norbit
                delivered=instlumi*lslength
                result[runnum]=[delivered,0.0,0.0]
            #result[runnum]=[0.0,0.0,0.0]
            #if c.VERBOSE: print 'request run ',runnum,' has no trigger, skip'
        else:
            norbits=lumitrginfo.values()[0][1]
            lslength=t.bunchspace_s*t.nbx*norbits
            delivered=totallumi*lslength
            hlttrgmap={}
            trgbitinfo={}
            if len(hltpath)!=0 and hltpath!='all':
                q=dbsession.nominalSchema().newQuery() #optional q3, initiated only if you ask for a hltpath
                hlttrgmap=lumiQueryAPI.hlttrgMappingByrun(q,runnum)
                del q
                if hlttrgmap.has_key(hltpath):
                    l1bitname=hltTrgSeedMapper.findUniqueSeed(hltpath,hlttrgmap[hltpath])
                    q=dbsession.nominalSchema().newQuery() #optional q4, initiated only if you ask for a hltpath and it exists 
                    hltinfo=lumiQueryAPI.hltBypathByrun(q,runnum,hltpath)
                    del q
                    q=dbsession.nominalSchema().newQuery()
                    trgbitinfo=lumiQueryAPI.trgBybitnameByrun(q,runnum,l1bitname) #optional q5, initiated only if you ask for a hltpath and it has a unique l1bit
                    del q
        #done all possible queries. process result
            for cmslsnum,valuelist in lumitrginfo.items():
                if len(selectionDict)!=0 and not (cmslsnum in selectionDict[runnum]):
                    #if there's a selection list but cmslsnum is not selected,skip
                    continue
                if valuelist[5]==0:#bitzero==0 means no beam,do nothing
                    continue
                trgprescale=valuelist[8]            
                deadfrac=float(valuelist[6])/float(float(valuelist[5])*float(trgprescale))
                if(deadfrac<1.0):
                    recorded=recorded+valuelist[0]*(1.0-deadfrac)*lslength
                    if hlttrgmap.has_key(hltpath) and hltinfo.has_key(cmslsnum):
                        hltprescale=hltinfo[cmslsnum][2]
                        trgprescale=trgbitinfo[cmslsnum][3]
                        recordedinpath=recordedinpath+valuelist[0]*(1.0-deadfrac)*lslength*hltprescale*trgprescale
                else:
                    if deadfrac<0.0:
                        print 'warning deadfraction negative in run',runnum,' ls ',cmslsnum
                if c.VERBOSE:
                    print runnum,cmslsnum,valuelist[0]*lslength,valuelist[0]*(1.0-deadfrac)*lslength,lslength,deadfrac
            result[runnum]=[delivered,recorded,recordedinpath]
    dbsession.transaction().commit()
    #if c.VERBOSE:
    #    print result
    return result           

def main():
    allowedscales=['linear','log','both']
    c=constants()
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Plot integrated luminosity as function of the time variable of choice",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # add required arguments
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    # add optional arguments
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file')
    parser.add_argument('-n',dest='normfactor',action='store',help='normalization factor (optional, default to 1.0)')
    parser.add_argument('-i',dest='inputfile',action='store',help='lumi range selection file (optional)')
    parser.add_argument('-o',dest='outputfile',action='store',help='csv outputfile name (optional)')
    parser.add_argument('-lumiversion',dest='lumiversion',default='0001',action='store',required=False,help='lumi data version')
    parser.add_argument('-begin',dest='begin',action='store',help='begin value of x-axi (required)')
    parser.add_argument('-end',dest='end',action='store',help='end value of x-axi (optional). Default to the maximum exists DB')
    parser.add_argument('-beamenergy',dest='beamenergy',action='store',type=float,required=False,help='beamenergy (in GeV) selection criteria,e.g. 3.5e3')
    parser.add_argument('-beamfluctuation',dest='beamfluctuation',action='store',type=float,required=False,help='allowed fraction of beamenergy to fluctuate, e.g. 0.1')
    parser.add_argument('-beamstatus',dest='beamstatus',action='store',required=False,help='selection criteria beam status,e.g. STABLE BEAMS')
    parser.add_argument('-yscale',dest='yscale',action='store',required=False,default='linear',help='y_scale')
    parser.add_argument('-hltpath',dest='hltpath',action='store',help='specific hltpath to calculate the recorded luminosity. If specified aoverlays the recorded luminosity for the hltpath on the plot')
    parser.add_argument('-batch',dest='batch',action='store',help='graphical mode to produce PNG file. Specify graphical file here. Default to lumiSum.png')
    parser.add_argument('--annotateboundary',dest='annotateboundary',action='store_true',help='annotate boundary run numbers')
    parser.add_argument('--interactive',dest='interactive',action='store_true',help='graphical mode to draw plot in a TK pannel.')
    parser.add_argument('-timeformat',dest='timeformat',action='store',help='specific python timeformat string (optional).  Default mm/dd/yy hh:min:ss.00')
    parser.add_argument('-siteconfpath',dest='siteconfpath',action='store',help='specific path to site-local-config.xml file, default to $CMS_PATH/SITECONF/local/JobConfig, if path undefined, fallback to cern proxy&server')
    parser.add_argument('action',choices=['run','fill','time','perday'],help='x-axis data type of choice')
    #graphical mode options
    parser.add_argument('--verbose',dest='verbose',action='store_true',help='verbose mode, print result also to screen')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    # parse arguments
    batchmode=True
    args=parser.parse_args()
    connectstring=args.connect
    begvalue=args.begin
    endvalue=args.end
    beamstatus=args.beamstatus
    beamenergy=args.beamenergy
    beamfluctuation=args.beamfluctuation
    xaxitype='run'
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
    hltpath=''
    if args.hltpath:
        hltpath=args.hltpath
    if args.debug :
        msg=coral.MessageStream('')
        msg.setMsgVerbosity(coral.message_Level_Debug)
    ifilename=''
    ofilename='integratedlumi.png'
    timeformat=''
    if args.authpath and len(args.authpath)!=0:
        os.environ['CORAL_AUTH_PATH']=args.authpath
    if args.normfactor:
        c.NORM=float(args.normfactor)
    if args.lumiversion:
        c.LUMIVERSION=args.lumiversion
    if args.verbose:
        c.VERBOSE=True
    if args.inputfile:
        ifilename=args.inputfile
    if args.batch:
        opicname=args.batch
    if args.outputfile:
        ofilename=args.outputfile
    if args.timeformat:
        timeformat=args.timeformat
    session=svc.connect(connectstring,accessMode=coral.access_Update)
    session.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")
    session.typeConverter().setCppTypeForSqlType("unsigned long long","NUMBER(20)")
    inputfilecontent=''
    fileparsingResult=''
    runList=[]
    runDict={}
    fillDict={}
    selectionDict={}
    minTime=''
    maxTime=''

    #if len(ifilename)!=0 :
    #    f=open(ifilename,'r')
    #    inputfilecontent=f.read()
    #    sparser=selectionParser.selectionParser(inputfilecontent)
    #    runsandls=sparser.runsandls()
    #    keylist=runsandls.keys()
    #    keylist.sort()
    #    for run in keylist:
    #        if selectionDict.has_key(run):
    #            lslist=runsandls[run]
    #            lslist.sort()
    #            selectionDict[run]=lslist
    if len(ifilename)!=0:
        ifparser=inputFilesetParser.inputFilesetParser(ifilename)
        runsandls=ifparser.runsandls()
        keylist=runsandls.keys()
        keylist.sort()
        for run in keylist:
            if not selectionDict.has_key(run):
                lslist=runsandls[run]
                lslist.sort()
                selectionDict[run]=lslist
    if args.action == 'run':
        if not args.end:
            session.transaction().start(True)
            schema=session.nominalSchema()
            lastrun=max(lumiQueryAPI.allruns(schema,requireRunsummary=True,requireLumisummary=True,requireTrg=True,requireHlt=True))
            session.transaction().commit()
        else:
            lastrun=int(args.end)
        for r in range(int(args.begin),lastrun+1):
            runList.append(r)
    elif args.action == 'fill':
        session.transaction().start(True)
        maxfill=None
        if not args.end:
            qHandle=session.nominalSchema().newQuery()
            maxfill=max(lumiQueryAPI.allfills(qHandle,filtercrazy=True))
            del qHandle
        else:
            maxfill=int(args.end)
        qHandle=session.nominalSchema().newQuery()
        fillDict=lumiQueryAPI.runsByfillrange(qHandle,int(args.begin),maxfill)
        del qHandle
        session.transaction().commit()
        #print 'fillDict ',fillDict
        for fill in range(int(args.begin),maxfill+1):
            if fillDict.has_key(fill): #fill exists
                for run in fillDict[fill]:
                    runList.append(run)
    elif args.action == 'time' or args.action == 'perday':
        session.transaction().start(True)
        t=lumiTime.lumiTime()
        minTime=t.StrToDatetime(args.begin,timeformat)
        if not args.end:
            maxTime=datetime.datetime.utcnow() #to now
        else:
            maxTime=t.StrToDatetime(args.end,timeformat)
        #print minTime,maxTime
        qHandle=session.nominalSchema().newQuery()
        runDict=lumiQueryAPI.runsByTimerange(qHandle,minTime,maxTime)#xrawdata
        session.transaction().commit()
        runList=runDict.keys()
        del qHandle
        #print runDict
    else:
        print 'unsupported action ',args.action
        exit
    runList.sort()
    #print 'runList ',runList
    #print 'runDict ', runDict
    
    fig=Figure(figsize=(6,4.5),dpi=100)
    m=matplotRender.matplotRender(fig)
    
    logfig=Figure(figsize=(6.8,4.5),dpi=100)
    mlog=matplotRender.matplotRender(logfig)
    
    if args.action == 'run':
        result={}        
        result=getLumiInfoForRuns(session,c,runList,selectionDict,hltpath,beamstatus=beamstatus,beamenergy=beamenergy,beamfluctuation=beamfluctuation)
        xdata=[]
        ydata={}
        ydata['Delivered']=[]
        ydata['Recorded']=[]
        keylist=result.keys()
        keylist.sort() #must be sorted in order
        if args.outputfile:
            reporter=csvReporter.csvReporter(ofilename)
            fieldnames=['run','delivered','recorded']
            reporter.writeRow(fieldnames)
        for run in keylist:
            xdata.append(run)
            delivered=result[run][0]
            recorded=result[run][1]
            ydata['Delivered'].append(delivered)
            ydata['Recorded'].append(recorded)
            if args.outputfile and (delivered!=0 or recorded!=0):
                reporter.writeRow([run,result[run][0],result[run][1]])                
        m.plotSumX_Run(xdata,ydata,yscale='linear')
        mlog.plotSumX_Run(xdata,ydata,yscale='log')
    elif args.action == 'fill':        
        lumiDict={}
        lumiDict=getLumiInfoForRuns(session,c,runList,selectionDict,hltpath,beamstatus=beamstatus,beamenergy=beamenergy,beamfluctuation=beamfluctuation)
        xdata=[]
        ydata={}
        ydata['Delivered']=[]
        ydata['Recorded']=[]
        #keylist=lumiDict.keys()
        #keylist.sort()
        if args.outputfile:
            reporter=csvReporter.csvReporter(ofilename)
            fieldnames=['fill','run','delivered','recorded']
            reporter.writeRow(fieldnames)
        fills=fillDict.keys()
        fills.sort()
        for fill in fills:
            runs=fillDict[fill]
            runs.sort()
            for run in runs:
                xdata.append(run)
                ydata['Delivered'].append(lumiDict[run][0])
                ydata['Recorded'].append(lumiDict[run][1])
                if args.outputfile :
                    reporter.writeRow([fill,run,lumiDict[run][0],lumiDict[run][1]])   
        #print 'input fillDict ',len(fillDict.keys()),fillDict
        m.plotSumX_Fill(xdata,ydata,fillDict,yscale='linear')
        mlog.plotSumX_Fill(xdata,ydata,fillDict,yscale='log')
    elif args.action == 'time' : 
        lumiDict={}
        lumiDict=getLumiInfoForRuns(session,c,runList,selectionDict,hltpath,beamstatus=beamstatus,beamenergy=beamenergy,beamfluctuation=beamfluctuation)
        #lumiDict=getLumiInfoForRuns(session,c,runList,selectionDict,hltpath,beamstatus='STABLE BEAMS')
        xdata={}#{run:[starttime,stoptime]}
        ydata={}
        ydata['Delivered']=[]
        ydata['Recorded']=[]
        keylist=lumiDict.keys()
        keylist.sort()
        if args.outputfile:
            reporter=csvReporter.csvReporter(ofilename)
            fieldnames=['run','starttime','stoptime','delivered','recorded']
            reporter.writeRow(fieldnames)
        for run in keylist:
            ydata['Delivered'].append(lumiDict[run][0])
            ydata['Recorded'].append(lumiDict[run][1])
            starttime=runDict[run][0]
            stoptime=runDict[run][1]
            xdata[run]=[starttime,stoptime]
            if args.outputfile :
                reporter.writeRow([run,starttime,stoptime,lumiDict[run][0],lumiDict[run][1]])
        m.plotSumX_Time(xdata,ydata,minTime,maxTime,hltpath=hltpath,annotateBoundaryRunnum=args.annotateboundary,yscale='linear')
        mlog.plotSumX_Time(xdata,ydata,minTime,maxTime,hltpath=hltpath,annotateBoundaryRunnum=args.annotateboundary,yscale='log')
    elif args.action == 'perday':
        daydict={}#{day:[[run,cmslsnum,lsstarttime,delivered,recorded]]}
        lumibyls=getLumiOrderByLS(session,c,runList,selectionDict,hltpath,beamstatus=beamstatus,beamenergy=beamenergy,beamfluctuation=beamfluctuation)
        #print 'lumibyls ',lumibyls
        #lumibyls [[runnumber,runstarttime,lsnum,lsstarttime,delivered,recorded,recordedinpath]]
        if args.outputfile:
            reporter=csvReporter.csvReporter(ofilename)
            fieldnames=['day','begrunls','endrunls','delivered','recorded']
            reporter.writeRow(fieldnames)
        beginfo=[lumibyls[0][3],str(lumibyls[0][0])+':'+str(lumibyls[0][2])]
        endinfo=[lumibyls[-1][3],str(lumibyls[-1][0])+':'+str(lumibyls[-1][2])]
        for perlsdata in lumibyls:
            lsstarttime=perlsdata[3]
            delivered=perlsdata[4]
            recorded=perlsdata[5]
            day=lsstarttime.toordinal()
            if not daydict.has_key(day):
                daydict[day]=[]
            daydict[day].append([delivered,recorded])
        days=daydict.keys()
        days.sort()
        daymin=days[0]
        daymax=days[-1]
        #alldays=range(daymin,daymax+1)
        resultbyday={}
        resultbyday['Delivered']=[]
        resultbyday['Recorded']=[]
        #for day in days:
        #print 'day min ',daymin
        #print 'day max ',daymax
        for day in range(daymin,daymax+1):
            if not daydict.has_key(day):
                delivered=0.0
                recorded=0.0
            else:
                daydata=daydict[day]
                mytransposed=CommonUtil.transposed(daydata,defaultval=0.0)
                delivered=sum(mytransposed[0])
                recorded=sum(mytransposed[1])
            resultbyday['Delivered'].append(delivered)
            resultbyday['Recorded'].append(recorded)
            if args.outputfile:
                reporter.writeRow([day,beginfo[1],endinfo[1],delivered,recorded])
        #print 'beginfo ',beginfo
        #print 'endinfo ',endinfo
        #print resultbyday
        m.plotPerdayX_Time( range(daymin,daymax+1) ,resultbyday,minTime,maxTime,boundaryInfo=[beginfo,endinfo],annotateBoundaryRunnum=args.annotateboundary,yscale='linear')
        mlog.plotPerdayX_Time( range(daymin,daymax+1),resultbyday,minTime,maxTime,boundaryInfo=[beginfo,endinfo],annotateBoundaryRunnum=args.annotateboundary,yscale='log')
    else:
        raise Exception,'must specify the type of x-axi'

    del session
    del svc

    if args.batch and args.yscale=='linear':
        m.drawPNG(args.batch)
    elif args.batch and args.yscale=='log':
        mlog.drawPNG(args.batch)
    elif args.batch and args.yscale=='both':
        m.drawPNG(args.batch)
        basename,extension=os.path.splitext(args.batch)
        logfilename=basename+'_log'+extension        
        mlog.drawPNG(logfilename)
    else:
        raise Exception('unsupported yscale for batch mode : '+args.yscale)
    if not args.interactive:
        return
    if args.interactive is True and args.yscale=='linear':
        m.drawInteractive()
    elif args.interactive is True and args.yscale=='log':
        mlog.drawInteractive()
    else:
        raise Exception('unsupported yscale for interactive mode : '+args.yscale)
if __name__=='__main__':
    main()
