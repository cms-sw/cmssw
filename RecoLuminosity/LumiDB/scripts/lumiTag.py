#!/usr/bin/env python

###################################################################
# Command to manage/display tags and dataids in lumiDB            #
#                                                                 #
# Author:      Zhen Xie                                           #
###################################################################

import os,sys
from RecoLuminosity.LumiDB import revisionDML,argparse,sessionManager,lumiReport

##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################


if __name__ == '__main__':
    parser=argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Lumi Normalization factor",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    allowedActions=['create','list']
    parser.add_argument('action',choices=allowedActions,help='command actions')
    parser.add_argument('-c',
                        dest='connect',
                        action='store',
                        required=False,
                        default='frontier://LumiCalc/CMS_LUMI_PROD',
                        help='connect string to lumiDB,optional'
                        )
    parser.add_argument('-P',
                        dest='authpath',
                        action='store',
                        help='path to authentication file,optional'
                        )
    parser.add_argument('--name',
                        dest='name',
                        action='store',
                        help='lumi datatag name'
                        )
    parser.add_argument('--lumitype',
                        dest='lumitype',
                        action='store',
                        required=True,
                        help='lumitype,HF,PIXEL'
                        )
    parser.add_argument('--siteconfpath',
                        dest='siteconfpath',
                        action='store',
                        help='specific path to site-local-config.xml file, optional. If path undefined, fallback to cern proxy&server'
                        )
    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        help='debug'
                        )
    options=parser.parse_args()

    if options.authpath:
        os.environ['CORAL_AUTH_PATH']=options.authpath 
    svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,siteconfpath=options.siteconfpath)
    session=None
    if options.action=='create':
        if not options.name:
            print '--name option is required'
            sys.exit(0)
        session=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])               
        session.transaction().start(False)
        schema=session.nominalSchema()
        revisionDML.createDataTag(schema,options.name,lumitype=options.lumitype)
        session.transaction().commit()
    if options.action=='list':
        session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])               
        session.transaction().start(True)
        if not options.name:
            alltags=revisionDML.alldataTags(session.nominalSchema(),lumitype=options.lumitype)
            lumiReport.toScreenTags(alltags)
        else:
            taginfo=revisionDML.dataIdsByTagName(session.nominalSchema(),options.name,runlist=None,lumitype=options.lumitype,withcomment=True)
            lumiReport.toScreenSingleTag(taginfo)
        session.transaction().commit()
    del session
    del svc
