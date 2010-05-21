#!/usr/bin/env python
VERSION='1.00'
import os,sys
import coral
from RecoLuminosity.LumiDB import argparse,nameDealer,selectionParser,hltTrgSeedMapper,connectstrParser,cacheconfigParser,matplotRender
from matplotlib.figure import Figure
class constants(object):
    def __init__(self):
        self.NORM=1.0
        self.LUMIVERSION='0001'
        self.NBX=3564
        self.BEAMMODE='stable' #possible choices stable,quiet,either

    def defaultfrontierConfigString(self):
        return """<frontier-connect><proxy url="http://cmst0frontier.cern.ch:3128"/><proxy url="http://cmst0frontier.cern.ch:3128"/><proxy url="http://cmst0frontier1.cern.ch:3128"/><proxy url="http://cmst0frontier2.cern.ch:3128"/><server url="http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier1.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier2.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier3.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier4.cern.ch:8000/FrontierInt"/></frontier-connect>"""
    
def plot(deliveredDict,recordedDict,hltpath=''):
    
    return fig

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
    parser.add_argument('action',choices=['run','fill','time'],help='x-axis data type of choice')
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
    if args.debug :
        msg=coral.MessageStream('')
        msg.setMsgVerbosity(coral.message_Level_Debug)
    hpath=''
    ifilename=''
    ofilename='integlumi.png'
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
    if len(ifilename)!=0 :
        f=open(ifilename,'r')
        inputfilecontent=f.read()
        fileparsingResult=selectionParser.selectionParser(inputfilecontent)

    fig=Figure(figsize=(5,4),dpi=100)
    a=fig.add_subplot(111)
    timedata=[]
    lumidata=[]
    if not args.action or args.action == 'run':
        timedata=[1,2,3,4]
        lumidata=[5,6,7,8]
    a.plot(timedata,lumidata)
    m=matplotRender.matplotRender(fig)
    if args.interactive:
        m.drawInteractive()
    else:
        m.drawPNG(ofilename)
    
    del session
    del svc
if __name__=='__main__':
    main()
