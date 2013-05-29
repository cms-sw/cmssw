#!/usr/bin/env python
import os,sys
from RecoLuminosity.LumiDB import dataDML,revisionDML,argparse,sessionManager

##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################


if __name__ == '__main__':
    parser=argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Lumi Normalization factor",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    allowedActions=['add','get','overview']
    amodetagChoices=['PROTPHYS','HIPHYS']
    egevChoices=['3500','450']
    parser.add_argument('action',choices=allowedActions,help='command actions')
    parser.add_argument('-c',dest='connect',action='store',required=False,help='connect string to lumiDB,optional',default='frontier://LumiCalc/CMS_LUMI_PROD')
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file,optional')
    parser.add_argument('-name',dest='name',action='store',help='lumi norm factor name')
    parser.add_argument('-egev',dest='egev',action='store',default='3500',choices=egevChoices,help='nominal beam energy in GeV')
    parser.add_argument('-amodetag',dest='amodetag',action='store',default='PROTPHYS',choices=amodetagChoices,help='accelerator mode')
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
    #elif options.action=='createBranch':
    #    #
    #    # create norm branch
    #    #        
    #    if not options.authpath:
    #        raise 'argument -P authpath is required for createBranch action'
    #    if not options.name:
    #        raise  RuntimeError('argument -name name is required for createBranch action')
    #    session=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')]) 
    #    session.transaction().start(False)
    #    schema=session.nominalSchema()
    #    trunk_rev_id,trunk_branch_id=revisionDML.branchInfoByName(schema,'TRUNK')
    #    if not trunk_rev_id and not trunk_branch_id:
    #        revisionDML.createBranch(schema,'TRUNK',None,comment='main')
    #    revisionDML.createBranch(schema,'NORM','TRUNK',comment='hold normalization factor')
    #    session.transaction().commit()
    elif options.action=='get':
        session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
        session.transaction().start(True)
        schema=session.nominalSchema()
        if options.tag is None:#ask for the latest of name
            if options.name is not None:
                normdataid=dataDML.guessnormIdByName(schema,options.name)
                norm=dataDML.luminormById(schema,normdataid)
                print 'norm for ',options.name,norm
            else:
                normdataid=dataDML.guessnormIdByContext(schema,options.amodetag,int(options.egev))
                norm=dataDML.luminormById(schema,normdataid)
                print 'norm for ',norm
        session.transaction().commit()
    elif options.action=='overview':
        session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
        session.transaction().start(True)
        schema=session.nominalSchema()
        branchfilter=revisionDML.revisionsInBranchName(schema,'NORM')
        allnorms=dataDML.mostRecentLuminorms(schema,branchfilter)
        print allnorms
        session.transaction().commit()
    del session
    del svc
