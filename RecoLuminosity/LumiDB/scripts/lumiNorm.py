#!/usr/bin/env python
import os,sys
from RecoLuminosity.LumiDB import dataDML,revisionDML,argparse,sessionManager,lumiReport

##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################


if __name__ == '__main__':
    parser=argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Lumi Normalization factor",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    allowedActions=['add','list']
    amodetagChoices=['PROTPHYS','IONPHYS']
    egevChoices=['3500','450','1380']
    parser.add_argument('action',choices=allowedActions,help='command actions')
    parser.add_argument('-c',dest='connect',action='store',required=False,help='connect string to lumiDB,optional',default='frontier://LumiCalc/CMS_LUMI_PROD')
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file,optional')
    parser.add_argument('-name',dest='name',action='store',help='lumi norm factor name')
    parser.add_argument('-egev',dest='egev',action='store',default=None,help='single beam energy in GeV')
    parser.add_argument('-amodetag',dest='amodetag',action='store',default=None,choices=amodetagChoices,help='accelerator mode')
    parser.add_argument('-input',dest='input',action='store',help='input lumi value. Option for add action only')
    parser.add_argument('-tag',dest='tag',action='store',help='version of the norm factor')
    parser.add_argument('-siteconfpath',dest='siteconfpath',action='store',help='specific path to site-local-config.xml file, optional. If path undefined, fallback to cern proxy&server')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    options=parser.parse_args()
    if options.authpath:
        os.environ['CORAL_AUTH_PATH']=options.authpath
    #
    #pre-check
    #
    if options.action=='add':
        if not options.authpath:
            raise RuntimeError('argument -P authpath is required for add action')
        if not options.input:
            raise RuntimeError('argument -input input is required for add action')
        if not options.name:
            raise RuntimeError('argument -name name is required for add action')
    svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,siteconfpath=options.siteconfpath)
    session=None
    
    if options.action=='add':
        session=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])        
        #
        # add norm factor
        #
        session.transaction().start(False)
        schema=session.nominalSchema()
        (revision_id,branch_id)=revisionDML.branchInfoByName(schema,'NORM')
        dataDML.addNormToBranch(schema,options.name,options.amodetag,float(options.input),int(options.egev),{},(revision_id,'NORM'))
        session.transaction().commit()
    elif options.action=='list':
        session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
        session.transaction().start(True)
        schema=session.nominalSchema()
        if options.tag is None:#ask for the latest of name
            if options.name is None and options.amodetag is None and options.egev is None:
                branchfilter=revisionDML.revisionsInBranchName(schema,'NORM')
                allnorms=dataDML.mostRecentLuminorms(schema,branchfilter)
                lumiReport.toScreenNorm(allnorms)
            elif options.name is not None:
                normdataid=dataDML.guessnormIdByName(schema,options.name)
                norm=dataDML.luminormById(schema,normdataid)
                nname=norm[0]
                namodetag=norm[1]
                nnormval=norm[2]
                negev=norm[3]
                lumiReport.toScreenNorm({nname:[namodetag,nnormval,negev]})
            else:
                amodetag=options.amodetag 
                if options.amodetag is None:
                    print '[Warning] -amodetag is not specified, assume PROTPHYS'
                    amodetag='PROTPHYS'
                egev=options.egev
                if options.egev is None:
                    print '[Warning] -egev is not specified, assume 3500'
                    egev='3500'
                normdataid=dataDML.guessnormIdByContext(schema,amodetag,int(egev))
                norm=dataDML.luminormById(schema,normdataid)
                nname=norm[0]
                namodetag=norm[1]
                nnormval=norm[2]
                negev=norm[3]
                lumiReport.toScreenNorm({nname:[namodetag,nnormval,negev]})
        session.transaction().commit()
    del session
    del svc
