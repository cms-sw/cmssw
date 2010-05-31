#!/usr/bin/env python
VERSION='1.00'
import os,sys
import coral
from RecoLuminosity.LumiDB import argparse,nameDealer,selectionParser,hltTrgSeedMapper,connectstrParser,cacheconfigParser,matplotRender
from matplotlib.figure import Figure
def findInList(mylist,element):
    pos=-1
    try:
        pos=mylist.index(element)
    except ValueError:
        pos=-1
    return pos!=-1
class constants(object):
    def __init__(self):
        self.NORM=1.0
        self.LUMIVERSION='0001'
        self.NBX=3564
        self.BEAMMODE='stable' #possible choices stable,quiet,either

    def defaultfrontierConfigString(self):
        return """<frontier-connect><proxy url="http://cmst0frontier.cern.ch:3128"/><proxy url="http://cmst0frontier.cern.ch:3128"/><proxy url="http://cmst0frontier1.cern.ch:3128"/><proxy url="http://cmst0frontier2.cern.ch:3128"/><server url="http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier1.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier2.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier3.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier4.cern.ch:8000/FrontierInt"/></frontier-connect>"""
    
def getLumiInfoForRuns(dbsession,c,runDict,hltpath=''):
    '''
    output:{ runnumber:[delivered,recorded,recorded_hltpath] }
    '''
    result={}#runnumber:[lumisumoverlumils,lumisumovercmsls-deadtimecorrected,lumisumovercmsls-deadtimecorrected*hltcorrection_hltpath]
    deliveredDict={}
    recordedPerRun={}
    prescaledRecordedPerRun={}
    #
    #delivered: select runnum,instlumi,numorbit from lumisummary where runnum>=:runMin and runnum<=runMax and lumiversion=:lumiversion order by runnum
    #calculate lslength = numorbit*numbx*25e-09
    #recorded: need instlumi and deadtime
    #select lumisummary.cmslsnum,lumisummary.runnum,lumisummary.instlumi,trg.deadtime,trg.trgcount from lumisummary,trg where lumisummary.runnum=trg.runnum and trg.cmslsnum=lumisummary.cmslsnum and trg.bitnum=0 and lumisummary.lumiversion='0001' and lumisummary.runnum>=133511 and lumisummary.runnum<=133877 order by lumisummary.runnum,lumisummary.cmslsnum;
    #          if hltpath is specified, need prescale L1 and hlt
    #          select trg.runnum,trg.bitnum,trg.bitname,trg.prescale,trghltmap.hltpathname,trghltgmap.l1seed,hlt.prescale from trg,trghltmap,hlt where hlt.hltpathname=:hltpathname and hlt.hltpathname=trghltmap.hltpathname and trg.cmslsnum=0 and hlt.runnum=trg.runnum and trghltmap.runnum=trg.runnum and trg.bitname=trghltmap.l1seed and trg.runnum>=:begRun and trg.runnum<=:endRun 
    try:
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        begRun=min(runDict.keys())
        endRun=max(runDict.keys())
        #delivered doesn't care about cmsls
        #print 'begRun ',begRun,'endRun ',endRun
        deliveredQuery=schema.tableHandle(nameDealer.lumisummaryTableName()).newQuery()
        deliveredQuery.addToOutputList('RUNNUM','runnum')
        deliveredQuery.addToOutputList('INSTLUMI','instlumi')
        deliveredQuery.addToOutputList('NUMORBIT','numorbit')
        
        deliveredQueryCondition=coral.AttributeList()
        deliveredQueryCondition.extend('begRun','unsigned int')
        deliveredQueryCondition.extend('endRun','unsigned int')
        deliveredQueryCondition.extend('lumiversion','string')
        deliveredQueryCondition['begRun'].setData(int(begRun))
        deliveredQueryCondition['endRun'].setData(int(endRun))
        deliveredQueryCondition['lumiversion'].setData(c.LUMIVERSION)
        deliveredQueryResult=coral.AttributeList()
        deliveredQueryResult.extend('runnum','unsigned int')
        deliveredQueryResult.extend('instlumi','float')
        deliveredQueryResult.extend('numorbit','unsigned int')
        deliveredQuery.defineOutput(deliveredQueryResult)
        deliveredQuery.setCondition('RUNNUM>=:begRun and RUNNUM<=:endRun and LUMIVERSION=:lumiversion',deliveredQueryCondition)
        deliveredQuery.addToOrderList('RUNNUM')
        deliveredQueryCursor=deliveredQuery.execute()
        #print 'runDict',runDict
        while deliveredQueryCursor.next():
            runnum=deliveredQueryCursor.currentRow()['runnum'].data()
            instlumi=deliveredQueryCursor.currentRow()['instlumi'].data()
            numorbit=deliveredQueryCursor.currentRow()['numorbit'].data()
            lslength=numorbit*c.NBX*25e-09
            #print runnum,instlumi,numorbit,lslength
            if runDict.has_key(runnum) and not deliveredDict.has_key(runnum):
                deliveredDict[runnum]=float(instlumi*lslength*c.NORM)
            elif runDict.has_key(runnum) and deliveredDict.has_key(runnum):
                deliveredDict[runnum]=deliveredDict[runnum]+float(instlumi*lslength*c.NORM)
        del deliveredQuery
        #print 'got delivered : ',deliveredDict
        
        lumiquery=schema.newQuery()
        lumiquery.addToTableList(nameDealer.lumisummaryTableName(),'lumisummary')
        lumiquery.addToTableList(nameDealer.trgTableName(),'trg')
        lumiqueryCondition=coral.AttributeList()
        lumiqueryCondition.extend('bitnum','unsigned int')
        lumiqueryCondition.extend('lumiversion','string')
        lumiqueryCondition.extend('begrun','unsigned int')
        lumiqueryCondition.extend('endrun','unsigned int')
        lumiqueryCondition['bitnum'].setData(int(0))
        lumiqueryCondition['lumiversion'].setData(c.LUMIVERSION)
        lumiqueryCondition['begrun'].setData(int(begRun))
        lumiqueryCondition['endrun'].setData(int(endRun))
        lumiquery.setCondition('lumisummary.RUNNUM=trg.RUNNUM and trg.CMSLSNUM=lumisummary.CMSLSNUM and trg.BITNUM=:bitnum and lumisummary.LUMIVERSION=:lumiversion and lumisummary.RUNNUM>=:begrun and lumisummary.RUNNUM<=:endrun',lumiqueryCondition)
        lumiquery.addToOrderList('lumisummary.runnum')
        lumiquery.addToOrderList('lumisummary.cmslsnum')
        lumiquery.addToOutputList('lumisummary.cmslsnum','cmsls')
        lumiquery.addToOutputList('lumisummary.runnum','runnum')
        lumiquery.addToOutputList('lumisummary.instlumi','instlumi')
        lumiquery.addToOutputList('trg.deadtime','deadtime')
        lumiquery.addToOutputList('trg.trgcount','trgcount')
        lumiqueryResult=coral.AttributeList()
        lumiqueryResult.extend('cmsls','unsigned int')
        lumiqueryResult.extend('runnum','unsigned int')
        lumiqueryResult.extend('instlumi','float')
        lumiqueryResult.extend('deadtime','unsigned long long')
        lumiqueryResult.extend('trgcount','unsigned int')
        lumiquery.defineOutput(lumiqueryResult)
        lumiqueryCursor=lumiquery.execute()
        correctedlumiSum=0.0
        while lumiqueryCursor.next():
            cmsls=lumiqueryCursor.currentRow()['cmsls'].data()
            runnum=lumiqueryCursor.currentRow()['runnum'].data()
            instlumi=lumiqueryCursor.currentRow()['instlumi'].data()
            deadtime=lumiqueryCursor.currentRow()['deadtime'].data()
            bitzero=lumiqueryCursor.currentRow()['trgcount'].data()
            deadfraction=0.0
            if bitzero==0:
                deadfraction=1.0
            else:
                deadfraction=float(deadtime)/float(bitzero)
                
            if runDict.has_key(runnum) and len(runDict[runnum])==0:
                if not recordedPerRun.has_key(runnum):
                    recordedPerRun[runnum]=float(instlumi*(1.0-deadfraction)*lslength*c.NORM)
                else:
                    recordedPerRun[runnum]=float(recordedPerRun[runnum]+instlumi*(1.0-deadfraction)*lslength*c.NORM)
            elif runDict.has_key(runnum) and findInList(runDict[runnum],cmsls):
                if not recordedPerRun.has_key(runnum):
                    recordedPerRun[runnum]=float(instlumi*(1.0-deadfraction)*lslength*c.NORM)
                else:
                    recordedPerRun[runnum]=float(recordedPerRun[runnum]+instlumi*(1.0-deadfraction)*lslength*c.NORM)
        #print 'got recorded : ',recordedPerRun
        del lumiquery
        #print 'hltpath ',hltpath
        if len(hltpath)!=0 and hltpath!='all':
            #
            #select l1seed from trghltmap where trghltmap.hltpathname =:hltpathname
            #
            seedQuery=schema.newQuery()
            seedQuery.addToTableList(nameDealer.trghltMapTableName(),'trghltmap')
            seedQueryCondition=coral.AttributeList()
            seedQueryCondition.extend('hltpathname','string')
            seedQueryCondition['hltpathname'].setData(hltpath)
            seedQueryResult=coral.AttributeList()
            seedQueryResult.addToOutputList('l1seed','string')
            seedQuery.defineOutput(seedQueryResult)
            seedQueryCursor=seedQuery.execute()
            l1seed=''
            l1bitname=''
            while seedQueryCursor.next():
                l1seed=seedQueryCursor.currentRow()['l1seed'].data()
            if len(l1seed)!=0:
                l1bitname=hltTrgSeedMapper.findUniqueSeed(hltpath,l1seed)
            del seedQuery
            #
            #select runnum,prescale from hlt where pathname=:hltpathname and cmslsnum=1 and runnum>=:begRun and runnum<=:endRun order by runnum
            #
            hltprescQuery=schema.newQuery()
            hltprescQuery.addToTableList(nameDealer.trgTableName(),'trg')
            hltprescQuery.addToTableList(nameDealer.hltTableName(),'hlt')
            hltprescQueryCondition=coral.AttributeList()
            hltprescQueryCondition.extend('hltpathname','string')
            hltprescQueryCondition.extend('begRun','unsigned int')
            hltprescQueryCondition.extend('endRun','unsigned int')
            hltprescQueryCondition.extend('cmslsnum','unsigned int')
            hltprescQueryCondition['hltpathname'].setData( hltpath )
            hltprescQueryCondition['begRun'].setData( begRun )
            hltprescQueryCondition['endRun'].setData( endRun )
            hltprescQueryCondition['cmslsnum'].setData( 1 )
            hltprescQuery.setCondition('PATHNAME=:hltpathname and RUNNUM>=:begRun and RUNNUM<=:endRun and CMSLSNUM=:cmslsnum')
            hltprescQueryResult=coral.AttributeList()
            hltprescQueryResult.addToOutputList('RUNNUM','runnum')
            hltprescQueryResult.addToOutputList('PRESCALE','hltprescale')
            hltprescQuery.addToOrderList('RUNNUM')
            hltprescQueryCursor=hltprescQuery.execute()
            hltprescaleDict={}
            while hltprescQueryCursor.next():
                runnum=hltprescQueryCursor.currentRow()['runnum'].data()
                hltprescale=hltprescQueryCursor.currentRow()['hltprescale'].data()
                if runDict.has_key(runnum):
                    hltrescaleDict[runnum]=hltprescale
            #print 'got hlt pescale ',hltprescaleDict
            del hltprescQuery
            #
            #select runnum,bitnum,bitname,prescale from trg where cmslsnum=1 and bitname=:bitname and runnum>=:begRun and runnum<=:endRun order by runnum
            #
            if not l1bitname or len(l1bitname)==0:
                print 'no l1bit found for hltpath : ',hltpath,' setting trg prescale to 1'
                
            else:
                trgprescQuery=schema.newQuery()
                trgprescQuery.addToTableList(nameDealer.trgTableName(),'trg')
                trgprescQueryCondition=coral.AttributeList()
                trgprescQueryCondition.extend('cmslsnum','unsigned int')
                trgprescQueryCondition.extend('bitname','string')
                trgprescQueryCondition.extend('begRun','unsigned int')
                trgprescQueryCondition.extend('endRun','unsigned int')
                trgprescQuery.setCondition('CMSLSNUM=:cmslsnum and BITNAME=:bitname and RUNNUM>=:begRun and RUNNUM<=:endRun')
                trgprescQueryResult=coral.AttributeList()
                trgprescQueryResult.addToOutputList('RUNNUM','runnum')
                trgprescQueryResult.addToOutputList('BITNUM','bitnum')
                trgprescQueryResult.addToOutputList('PRESCALE','trgprescale')
                trgprescQuery.defineOutput(trgprescQueryResult)
                trgprescQueryCursor=trgprescQuery.execute()
                trgprescaleDict={}
                while trgprescQueryCursor.next():
                    runnum=trgprescQueryCursor.currentRow()['runnum'].data()
                    bitnum=trgprescQueryCursor.currentRow()['bitnum'].data()
                    trgprescale=trgprescQueryCursor.currentRow()['trgprescale'].data()
                    if runDict.has_key(runnum):
                        trgprescaleDict[runnum]=trgprescale
                del trgprescQuery
                #print trgprescaleDict
            for runnum,recorded in recordedPerRun.items():
                hltprescale=1.0
                trgprescale=1.0
                if hltprescQuery.has_key(runnum):
                    hltprescale=hltprescaleDict[runnum]
                if trgprescaleDict.has_key(runnum):
                    trgprescale=trgprescaleDict[runnum]
                prescaledRecordedPerRun[runnum]=recorded/(hltprescale*trgprescale)
        dbsession.transaction().commit()
        keylist=runDict.keys()
        keylist.sort()
        for runnum in keylist:
            if deliveredDict.has_key(runnum):
                result[runnum]=[]
                result[runnum].append(deliveredDict[runnum])
            else:
                result[runnum]=[]
                result[runnum].append(0)
            if recordedPerRun.has_key(runnum) and result.has_key(runnum):
                result[runnum].append(recordedPerRun[runnum])
            else:
                result[runnum].append(0)              
            if prescaledRecordedPerRun.has_key(runnum) and recordedPerRun.has_key(runnum) and result.has_key(runnum):
                result[runnum].append(prescaledRecordedPerRun[runnum])
            else:
                result[runnum].append(0)
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
    return result

def getRunsForFills(dbsession,minFill,maxFill):
    '''
    output:{ runnumber:[delivered,recorded,recorded_hltpath] }
    '''
    #find all runs in the fill range
    #select runnum,fillnum from cmsrunsummary where fillnum>=minFill and fillnum<=maxFill order by fillnum
    #
    fillDict={} #{fillnum:[runlist]}
    try:
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        fillQuery=schema.tableHandle(nameDealer.cmsrunsummaryTableName()).newQuery()
        fillQuery.addToOutputList('RUNNUM','runnum')
        fillQuery.addToOutputList('FILLNUM','fillnum')
        
        fillQueryCondition=coral.AttributeList()
        fillQueryCondition.extend('begFill','unsigned int')
        fillQueryCondition.extend('endFill','unsigned int')
        fillQueryCondition['begFill'].setData(int(minFill))
        fillQueryCondition['endFill'].setData(int(maxFill))
        fillQueryResult=coral.AttributeList()
        fillQueryResult.extend('runnum','unsigned int')
        fillQueryResult.extend('fillnum','unsigned int')

        fillQuery.defineOutput(fillQueryResult)
        fillQuery.setCondition('FILLNUM>=:begFill and FILLNUM<=:endFill',fillQueryCondition)
        fillQuery.addToOrderList('FILLNUM')
        fillQueryCursor=fillQuery.execute()
        while fillQueryCursor.next():
            runnum=fillQueryCursor.currentRow()['runnum'].data()
            fillnum=fillQueryCursor.currentRow()['fillnum'].data()
            if not fillDict.has_key(fillnum):
                fillDict[fillnum]=[runnum]
            else:
                fillDict[fillnum].append(runnum)
        del fillQuery
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
    return fillDict

def main():
    c=constants()
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Plot integrated luminosity as function of the time variable of choice")
    # add required arguments
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    # add optional arguments
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file')
    parser.add_argument('-n',dest='normfactor',action='store',help='normalization factor (optional, default to 1.0)')
    parser.add_argument('-i',dest='inputfile',action='store',help='lumi range selection file (optional)')
    parser.add_argument('-o',dest='outputfile',action='store',help='output PNG file (works with batch graphical mode, if not specified, default filename is instlumi.png)')
    parser.add_argument('-b',dest='beammode',action='store',help='beam mode, optional for delivered action, default "stable", choices "stable","quiet","either"')
    parser.add_argument('-lumiversion',dest='lumiversion',action='store',help='lumi data version, optional for all, default 0001')
    parser.add_argument('-begin',dest='begin',action='store',help='begin value of x-axi (required)')
    parser.add_argument('-end',dest='end',action='store',help='end value of x-axi (required)')
    parser.add_argument('-hltpath',dest='hltpath',action='store',help='specific hltpath to calculate the recorded luminosity. If specified aoverlays the recorded luminosity for the hltpath on the plot')
    parser.add_argument('-siteconfpath',dest='siteconfpath',action='store',help='specific path to site-local-config.xml file, default to $CMS_PATH/SITECONF/local/JobConfig, if path undefined, fallback to cern proxy&server')
    parser.add_argument('action',choices=['run','fill','date'],help='x-axis data type of choice')
    #graphical mode options
    parser.add_argument('--interactive',dest='interactive',action='store_true',help='graphical mode to draw plot in a TK pannel')
    parser.add_argument('--batch',dest='batch',action='store_true',help='graphical mode to produce PNG file only(default mode). Use -o option to specify the file name')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    # parse arguments
    args=parser.parse_args()
    connectstring=args.connect
    begvalue=args.begin
    endvalue=args.end
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
    beammode='stable'

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
    if args.outputfile and len(args.outputfile)!=0:
        ofilename=args.outputfile

    session=svc.connect(connectstring,accessMode=coral.access_Update)
    session.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")
    session.typeConverter().setCppTypeForSqlType("unsigned long long","NUMBER(20)")
    inputfilecontent=''
    fileparsingResult=''
    runDict={}
    fillDict={}
    if args.action == 'run':
        for r in range(int(args.begin),int(args.end)+1):
            runDict[r]=[]
    elif args.action == 'fill':
        fillDict=getRunsForFills(session,int(args.begin),int(args.end))
        #print 'fillDict ',fillDict
        for fill in range(int(args.begin),int(args.end)+1):
            if fillDict.has_key(fill): #fill exists
                for run in fillDict[fill]:
                    runDict[run]=[]
                
    if len(ifilename)!=0 :
            f=open(ifilename,'r')
            inputfilecontent=f.read()
            sparser=selectionParser.selectionParser(inputfilecontent)
            runsandls=sparser.runsandls()
            keylist=runsandls.keys()
            keylist.sort()
            for run in keylist:
                if runDict.has_key(run):
                    lslist=runsandls[run]
                    lslist.sort()
                    runDict[run]=lslist
    #print 'runDict ', runDict               
    fig=Figure(figsize=(5,4),dpi=100)
    m=matplotRender.matplotRender(fig)
    
    if args.action == 'run':
        result={}        
        result=getLumiInfoForRuns(session,c,runDict,hltpath)
        xdata=[]
        ydata={}
        ydata['Delivered']=[]
        ydata['Recorded']=[]
        keylist=result.keys()
        keylist.sort() #must be sorted in order
        for run in keylist:
            xdata.append(run)
            ydata['Delivered'].append(result[run][0])
            ydata['Recorded'].append(result[run][1])
        m.plotSumX_Run(xdata,ydata)
    elif args.action == 'fill':        
        lumiDict={}
        lumiDict=getLumiInfoForRuns(session,c,runDict,hltpath)
        xdata=[]
        ydata={}
        ydata['Delivered']=[]
        ydata['Recorded']=[]
        keylist=lumiDict.keys()
        keylist.sort()
        for run in keylist:
            xdata.append(run)
            ydata['Delivered'].append(lumiDict[run][0])
            ydata['Recorded'].append(lumiDict[run][1])
        m.plotSumX_Fill(xdata,ydata,fillDict)
    else:
        raise Exception,'must specify the type of x-axi'

    if args.interactive:
        m.drawInteractive()
    else:
        m.drawPNG(ofilename)
    
    del session
    del svc
if __name__=='__main__':
    main()
