import coral,os,os.path
from RecoLuminosity.LumiDB import cacheconfigParser,connectstrParser

class sessionManager(object):
    def defaultfrontierConfigString ():
        return '''<frontier-connect><proxy url = "http://cmst0frontier.cern.ch:3128"/><proxy url = "http://cmst0frontier.cern.ch:3128"/><proxy url = "http://cmst0frontier1.cern.ch:3128"/><proxy url = "http://cmst0frontier2.cern.ch:3128"/><server url = "http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url = "http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url = "http://cmsfrontier1.cern.ch:8000/FrontierInt"/><server url = "http://cmsfrontier2.cern.ch:8000/FrontierInt"/><server url = "http://cmsfrontier3.cern.ch:8000/FrontierInt"/><server url = "http://cmsfrontier4.cern.ch:8000/FrontierInt"/></frontier-connect>'''
    
    def __init__(self,connectString,authpath=None,siteconfpath=None,debugON = False):
        self.__connectString=connectString
        self.__svc=None
        self.__connectparser=connectstrParser.connectstrParser(self.__connectString)
        usedefaultfrontierconfig = False
        cacheconfigpath = ''
        try:
            self.__connectparser.parse()
            if self.__connectparser.needsitelocalinfo():
                if not siteconfpath:
                    cacheconfigpath = os.environ['CMS_PATH']
                    if cacheconfigpath:
                        cacheconfigpath = os.path.join (cacheconfigpath, 'SITECONF', 'local', 'JobConfig', 'site-local-config.xml')
                    else:
                        usedefaultfrontierconfig = True
                else:
                    cacheconfigpath = siteconfpath
                    cacheconfigpath = os.path.join (cacheconfigpath, 'site-local-config.xml')
                ccp = cacheconfigParser.cacheconfigParser()
                if usedefaultfrontierconfig:
                    ccp.parseString ( self.defaultfrontierConfigString() )
                else:
                    ccp.parse (cacheconfigpath)
                self.__connectString = self.__connectparser.fullfrontierStr(self.__connectparser.schemaname(), ccp.parameterdict())
            if self.__connectparser.protocol()=='oracle':
                if authpath:
                    os.environ['CORAL_AUTH_PATH']=authpath
                else:
                    os.environ['CORAL_AUTH_PATH']='.'
            if debugON :
                msg = coral.MessageStream ('')
                msg.setMsgVerbosity (coral.message_Level_Debug)            
            self.__svc = coral.ConnectionService()
        except:
            if self.__svc: del self.__svc
            raise
    def openSession(self,isReadOnly=True,cpp2sqltype=[],sql2cpptype=[] ):
        try:
            session=None
            if self.__connectparser.protocol()=='frontier' or isReadOnly:
                session = self.__svc.connect(self.__connectString, accessMode = coral.access_ReadOnly)
            else:
                session = self.__svc.connect(self.__connectString, accessMode = coral.access_Update)
            for (cpptype,sqltype) in cpp2sqltype:
                session.typeConverter().setCppTypeForSqlType(cpptype,sqltype)
            for (sqltype,cpptype) in sql2cpptype:
                session.typeConverter().setSqlTypeForCppType(sqltype,cpptype)
            return session
        except:
            if session: del session
            raise
    def realConnectString(self):
        return self.__connectString
    def __del__(self):
        del self.__svc
    def svcHandle(self):
        return self.__svc

if __name__ == "__main__":
    svc=sessionManager('oracle://cms_orcoff_prep/cms_lumi_dev_offline',authpath='/afs/cern.ch/user/x/xiezhen',debugON=False)
    print svc.realConnectString()
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    session.transaction().start(True)
    session.transaction().commit()
    del session
    svc=sessionManager('frontier://LumiPrep/CMS_LUMI_DEV_OFFLINE',debugON=False)
    print svc.realConnectString()
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    del session
