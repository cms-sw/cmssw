#!/usr/bin/env python

#########################################################################
# Command to manage func/version/validity of lumi corrections in lumiDB #
#                                                                       #
# Author:      Zhen Xie                                                 #
#########################################################################

import os,sys
from RecoLuminosity.LumiDB import normDML,revisionDML,argparse,sessionManager,lumiReport,normFileParser,CommonUtil

##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################

if __name__ == '__main__':
    parser=argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Luminosity normalization/correction management tool",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    allowedActions=['list','create','insert','setdefault','unsetdefault']
    parser.add_argument('action',choices=allowedActions,help='command actions')
    parser.add_argument('-c',
                        dest='connect',
                        action='store',
                        required=False,
                        help='connect string to lumiDB,optional',default='frontier://LumiCalc/CMS_LUMI_PROD')
    parser.add_argument('-P',
                        dest='authpath',
                        action='store',
                        help='path to authentication file,optional')
    parser.add_argument('--name',
                        dest='normname',
                        action='store',
                        help='norm name')
    parser.add_argument('--lumitype',
                        dest='lumitype',
                        action='store',
                        help='lumitype')
    parser.add_argument('-f',
                        dest='normfile',
                        action='store',
                        help='norm definition file. Required for all update actions')
    parser.add_argument('--siteconfpath',
                        dest='siteconfpath',
                        action='store',
                        help='specific path to site-local-config.xml file, optional. If path undefined, fallback to cern proxy&server')
    parser.add_argument('--firstsince',
                        dest='firstsince',
                        action='store',
                        default=None,
                        help='pick only the pieces with since>=firstsince to insert')
    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        help='debug')
    options=parser.parse_args()
    if options.authpath:
        os.environ['CORAL_AUTH_PATH']=options.authpath
    #############################################################
    #pre-check option compatibility
    #############################################################
    if options.action in ['create','insert','setdefault','unsetdefault']:
        if not options.connect:
            raise RuntimeError('argument -c connect is required for create/insert/updatedefault action')
        if not options.authpath:
            raise RuntimeError('argument -P authpath is required for create/insert/updatedefault action')
        if options.action in ['create','insert']:
            if not options.normfile:
                raise RuntimeError('argument -f normfile is required for insert action')
    if options.action in ['setdefault','unsetdefault']:
        if not options.lumitype:
            raise RuntimeError('argument --lumitype lumitype is required for setdefault/unsetdefault action')
        if not options.normname:
            raise RuntimeError('argument --name normname is required for setdefault/unsetdefault action')
    svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,siteconfpath=options.siteconfpath)
    ############################################################
    #  create,insert
    ############################################################
    if options.action in ['create','insert'] :
        dbsession=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
        normfileparser=normFileParser.normFileParser(options.normfile)
        normdata=normfileparser.parse()
        normdefinitionDict=normdata[0]#{defoption:value}
        normvalues=normdata[1]        #[{dataoption:value}]         
        dbsession.transaction().start(False)        
        normname=''
        if options.normname:#commandline has priorty
            normname=options.normname
        else:
            if normdefinitionDict.has_key('name') and normdefinitionDict['name']:
                normname=normdefinitionDict['name']
        if not normname:
            raise RuntimeError('[ERROR] normname undefined')
        lumitype='HF'
        if options.lumitype:
            lumitype=options.lumitype
        else:
            if normdefinitionDict.has_key('lumitype') and normdefinitionDict['lumitype']:
                lumitype=normdefinitionDict['lumitype']
        istypedefault=0
        if normdefinitionDict.has_key('istypedefault') and normdefinitionDict['istypedefault']:
            istypedefault=int(normdefinitionDict['istypedefault'])
        commentStr=''
        if normdefinitionDict.has_key('comment'):
            commentStr=normdefinitionDict['comment']
            
        if options.action=='create':
            (revision_id,branch_id)=revisionDML.branchInfoByName(dbsession.nominalSchema(),'NORM')
            branchinfo=(revision_id,'NORM')
            (normrev_id,normentry_id,normdata_id)=normDML.createNorm(dbsession.nominalSchema(),normname,lumitype,istypedefault,branchinfo,comment=commentStr)
        else:
            normdata_id=normDML.normIdByName(dbsession.nominalSchema(),normname)
        for normvalueDict in normvalues:
            if not normvalueDict.has_key('corrector') or not normvalueDict['corrector']:
                raise RuntimeError('parameter corrector is required for create/insert action')
            if not normvalueDict.has_key('since') or not normvalueDict['since']:
                raise RuntimeError('parameter since is required for create/insert action')
            correctorStr=normvalueDict['corrector']
            sincerun=int(normvalueDict['since'])
            if options.firstsince:
                if sincerun<int(options.firstsince):
                    continue
            amodetag=normvalueDict['amodetag']
            egev=int(normvalueDict['egev'])
            detailcomment=normvalueDict['comment']
            (correctorname,parameterlist)=CommonUtil.parselumicorrector(correctorStr)
            parameterDict={}
            for param in parameterlist:
                parameterDict[param]=normvalueDict[param]
            normDML.insertValueToNormId(dbsession.nominalSchema(),normdata_id,sincerun,correctorStr,amodetag,egev,parameterDict,comment=detailcomment)
        dbsession.transaction().commit()
        
    ##############################
    #  setdefault/unsetdefault
    ##############################
    if options.action in ['setdefault','unsetdefault']:
        dbsession=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
        dbsession.transaction().start(False)      
        if options.action=='setdefault':
            normDML.promoteNormToTypeDefault(dbsession.nominalSchema(),options.normname,options.lumitype)
        if options.action=='unsetdefault':
            normDML.demoteNormFromTypeDefault(dbsession.nominalSchema(),options.normname,options.lumitype)
        dbsession.transaction().commit()        
     ##############################
     #  list
     ##############################
    if options.action=='list':
        dbsession=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])      
        dbsession.transaction().start(True)
        if options.normname:
            norminfo=normDML.normInfoByName(dbsession.nominalSchema(),options.normname)
            normdataid=norminfo[0]
            normvalues=normDML.normValueById(dbsession.nominalSchema(),normdataid)
            lumiReport.toScreenNormDetail(options.normname,norminfo,normvalues)
        elif options.lumitype:
            luminormidmap=normDML.normIdByType(dbsession.nominalSchema(),lumitype=options.lumitype,defaultonly=False)
            for normname,normid in luminormidmap.items():
                norminfo=normDML.normInfoByName(dbsession.nominalSchema(),normname)
                normvalues=normDML.normValueById(dbsession.nominalSchema(),normid)
                lumiReport.toScreenNormDetail(normname,norminfo,normvalues)
        else:
            allnorms=normDML.allNorms(dbsession.nominalSchema())
            lumiReport.toScreenNormSummary(allnorms)
        dbsession.transaction().commit()
    del dbsession
    del svc
