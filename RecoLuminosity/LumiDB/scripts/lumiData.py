#!/usr/bin/env python
VERSION='2.00'
import os,sys
import coral
from RecoLuminosity.LumiDB import argparse,nameDealer,connectstrParser,cacheconfigParser,tablePrinter
from RecoLuminosity.LumiDB.wordWrappers import wrap_always,wrap_onspace,wrap_onspace_strict
def defaultfrontierConfigString(self):
        return """<frontier-connect><proxy url="http://cmst0frontier.cern.ch:3128"/><proxy url="http://cmst0frontier.cern.ch:3128"/><proxy url="http://cmst0frontier1.cern.ch:3128"/><proxy url="http://cmst0frontier2.cern.ch:3128"/><server url="http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier1.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier2.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier3.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier4.cern.ch:8000/FrontierInt"/></frontier-connect>"""
def listRemoveDuplicate(inlist):
    d={}
    for x in inlist:
        d[x]=x
    return d.values()

def runListInDB(dbsession,lumiversion=''):
    """
    list available runs in the DB
    output: [runnumber]
    """ 
    runlist=[]
    try:
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
         
        query=schema.tableHandle(nameDealer.lumisummaryTableName()).newQuery()
        query.addToOutputList("RUNNUM","run")
        query.addToOutputList("LUMIVERSION","lumiversion")
        queryBind=coral.AttributeList()
        queryBind.extend("lumiversion","string")
        if len(lumiversion)!=0:
            queryBind["lumiversion"].setData(lumiversion)
            query.setCondition("LUMIVERSION=:lumiversion",queryBind)
        query.addToOrderList('RUNNUM')
        result=coral.AttributeList()
        result.extend("run","unsigned int")
        result.extend("lumiversion","string")
        query.defineOutput(result)
        cursor=query.execute()
        while cursor.next():
            r=cursor.currentRow()['run'].data()
            v=cursor.currentRow()['lumiversion'].data()
            runlist.append((r,v))
        del query
        dbsession.transaction().commit()
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
    runlist=listRemoveDuplicate(runlist)
    runlist.sort()
    return runlist

def printRunList(runlistdata):
    result=[['Run','Lumiversion']]
    for tp in runlistdata:
        i=[str(tp[0]),tp[1]]
        result.append(i)
    print tablePrinter.indent(result,hasHeader=True,separateRows=False,prefix='| ',postfix=' |',wrapfunc=lambda x: wrap_onspace(x,20) )
    
def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Lumi Data operations")
    # add required arguments
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    # add optional arguments
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file')
    parser.add_argument('-lumiversion',dest='lumiversion',action='store',help='lumi data version, optional')
    parser.add_argument('-siteconfpath',dest='siteconfpath',action='store',help='specific path to site-local-config.xml file, default to $CMS_PATH/SITECONF/local/JobConfig, if path undefined, fallback to cern proxy&server')
    parser.add_argument('action',choices=['listrun'],help='command actions')
    parser.add_argument('--raw',dest='printraw',action='store_true',help='print raw data' )
    parser.add_argument('--verbose',dest='verbose',action='store_true',help='verbose mode for printing' )
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    # parse arguments
    args=parser.parse_args()
    connectstring=args.connect
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
    isverbose=False
    if args.debug :
        msg=coral.MessageStream('')
        msg.setMsgVerbosity(coral.message_Level_Debug)
        c.VERBOSE=True

    if args.verbose :
        c.VERBOSE=True
    if args.authpath and len(args.authpath)!=0:
        os.environ['CORAL_AUTH_PATH']=args.authpath

    session=svc.connect(connectstring,accessMode=coral.access_Update)
    session.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")

    if args.action == 'listrun':
        lumiversion=''
        if args.lumiversion:
            lumiversion=args.lumiversion
        runlist=runListInDB(session,lumiversion)
	if args.printraw:
	    print runlist
	else: printRunList(runlist)
    
    del session
    del svc
if __name__=='__main__':
    main()
    
