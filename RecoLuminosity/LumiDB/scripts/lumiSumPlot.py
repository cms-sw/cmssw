#!/usr/bin/env python
VERSION='1.00'
import os,sys
import coral
from RecoLuminosity.LumiDB import lumiTime,argparse,nameDealer,selectionParser,hltTrgSeedMapper,connectstrParser,cacheconfigParser,matplotRender,lumiQueryAPI
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
        self.VERBOSE=False
    def defaultfrontierConfigString(self):
        return """<frontier-connect><proxy url="http://cmst0frontier.cern.ch:3128"/><proxy url="http://cmst0frontier.cern.ch:3128"/><proxy url="http://cmst0frontier1.cern.ch:3128"/><proxy url="http://cmst0frontier2.cern.ch:3128"/><server url="http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier1.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier2.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier3.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier4.cern.ch:8000/FrontierInt"/></frontier-connect>"""

    
def getLumiInfoForRuns(dbsession,c,runDict,hltpath=''):
    '''
    input: runDict{runnum:[ls]}
    output:{ runnumber:[delivered,recorded,recorded_hltpath] }
    '''
    t=lumiTime.lumiTime()
    result={}#runnumber:[lumisumoverlumils,lumisumovercmsls-deadtimecorrected,lumisumovercmsls-deadtimecorrected*hltcorrection_hltpath]
    keylist=runDict.keys()
    keylist.sort()
    dbsession.transaction().start(True)
    for runnum in keylist:
        totallumi=0.0
        delivered=0.0
        recorded=0.0 
        recordedinpath=0.0
        #print 'looking for run ',runnum
        q=dbsession.nominalSchema().newQuery()
        totallumi=lumiQueryAPI.lumisumByrun(q,runnum,c.LUMIVERSION) #q1
        del q
        if not totallumi:
            result[runnum]=[0.0,0.0,0.0]
            if c.VERBOSE: print 'run ',runnum,' does not exist, skip'
            continue
        lumitrginfo={}
        hltinfo={}
        hlttrgmap={}
        q=dbsession.nominalSchema().newQuery()
        lumitrginfo=lumiQueryAPI.lumisummarytrgbitzeroByrun(q,runnum,c.LUMIVERSION) #q2
        del q
        if len(lumitrginfo)==0:
            result[runnum]=[0.0,0.0,0.0]
            if c.VERBOSE: print 'request run ',runnum,' has no trigger, skip'
            continue
        norbits=lumitrginfo[1][1]
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
            if valuelist[5]==0:#bitzero==0 means no beam,do nothing
                continue
            deadfrac=valuelist[6]/valuelist[5]
            trgprescale=valuelist[8]
            recorded=recorded+valuelist[0]*(1.0-deadfrac)*lslength
            if hlttrgmap.has_key(hltpath) and hltinfo.has_key(cmslsnum):
                hltprescale=hltinfo[cmslsnum][2]
                trgprescale=trgbitinfo[cmslsnum][3]
                recordedinpath=recordedinpath+valuelist[0]*(1.0-deadfrac)*lslength*hltprescale*trgprescale
        result[runnum]=[delivered,recorded,recordedinpath]
    dbsession.transaction().commit()
    if c.VERBOSE:
        print result
    return result           

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
    parser.add_argument('--verbose',dest='verbose',action='store_true',help='verbose mode, print result also to screen')
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
    if args.verbose:
        c.VERBOSE=True
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
        session.transaction().start(True)
        qHandle=session.nominalSchema().newQuery()
        fillDict=lumiQueryAPI.runsByfillrange(qHandle,int(args.begin),int(args.end))
        del qHandle
        session.transaction().commit()
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
    fig=Figure(figsize=(7,4),dpi=100)
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
        #m.plotSumX_Fill(xdata,ydata,fillDict)
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
