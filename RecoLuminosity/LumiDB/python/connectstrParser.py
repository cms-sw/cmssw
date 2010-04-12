import re
class connectstrParser(object):
    def __init__(self,connectstr):
        self.__connectstr=connectstr
        self.__pattern=re.compile('(^[\w]+)(://)(.+)(/)(.+)')
        self.__protocol=''
        self.__schemaname=''
        self.__servicename=''
        self.__servlettotranslate=''
    def parse(self):
        result=re.match(self.__pattern,self.__connectstr)
        if result is not None:
            protocol=result.group(1)
            if protocol!='frontier' and protocol!='oracle':
                raise 'unsupported technology',protocol
            self.__protocol=protocol
            self.__schemaname=result.group(5)
            self.__servicename=result.group(3)
            if self.__protocol=='frontier':
                if self.__servicename.find('(')==-1:
                    if self.__servicename.find('/')==-1:
                        self.__servlettotranslate=self.__servicename
                elif self.__servicename.find('/') == -1:
                    p=re.compile('([\w+])\(.+=.+\)')
                    r=re.search(p,self.__servicename)
                    if r is not None:
                        self.__servlettotranslate=r.group(1)
                    
    def protocol(self):
        return self.__protocol
    def schemaname(self):
        return self.__schemaname
    def service(self):
        return self.__servicename
    def needsitelocalinfo(self):
        if self.__protocol=='frontier':
            if len(self.__servlettotranslate)==0:
                return False
            else :
                return True
        else:
            return False
    def servlettotranslate(self):
        return self.__servlettotranslate
    def fullfrontierStr(self,schemaname,parameterDict):
        if len(parameterDict)==0:
            raise 'empty frontier parameters, cannot construct full connect string'
        result='frontier://'
        for k,v in parameterDict.items():
            ##if attr name=url, concatenate; if attrname=value discard
            for (attrname,attrvalue) in v:
                if attrname=='url':
                    result+='('+k+'url='+attrvalue+')'
                else:
                    result+='('+k+'='+attrvalue+')'
        result+='/'+schemaname
        return result
if __name__ == '__main__':
    import frontierconfigParser
    o='oracle://cms_orcoff_prep/CMS_LUMI_DEV_OFFLINE'
    parser=connectstrParser(o)
    parser.parse()
    print parser.protocol(),parser.service(),parser.schemaname(),parser.needsitelocalinfo()
    f1='frontier://cmsfrontier.cern.ch:8000/LumiPrep/CMS_LUMI_DEV_OFFLINE'
    parser=connectstrParser(f1)
    parser.parse()
    print parser.protocol(),parser.service(),parser.schemaname(),parser.needsitelocalinfo()
    f2='frontier://(serverurl=cmsfrontier.cern.ch:8000/LumiPrep/CMS_LUMI_DEV_OFFLINE)'
    parser=connectstrParser(f2)
    parser.parse()
    print parser.protocol(),parser.service(),parser.schemaname(),parser.needsitelocalinfo()
    f3='frontier://(proxyurl=http://cmst0frontier.cern.ch:3128)(proxyurl=http://cmst0frontier.cern.ch:3128)(proxyurl=http://cmst0frontier1.cern.ch:3128)(proxyurl=http://cmst0frontier2.cern.ch:3128)(serverurl=http://cmsfrontier.cern.ch:8000/LumiPrep)(serverurl=http://cmsfrontier.cern.ch:8000)/LumiPrep)(serverurl=http://cmsfrontier1.cern.ch:8000/LumiPrep)(serverurl=http://cmsfrontier2.cern.ch:8000/LumiPrep)(serverurl=http://cmsfrontier3.cern.ch:8000/LumiPrep)(serverurl=http://cmsfrontier4.cern.ch:8000/LumiPrep)/CMS_LUMI_DEV_OFFLINE'
    parser=connectstrParser(f3)
    parser.parse()
    print parser.protocol(),parser.service(),parser.schemaname(),parser.needsitelocalinfo()
    f4='frontier://LumiPrep/CMS_LUMI_DEV_OFFLINE'
    parser=connectstrParser(f4)
    parser.parse()
    print parser.protocol(),parser.service(),parser.schemaname(),parser.needsitelocalinfo()
    if parser.needsitelocalinfo():
        sitelocalconfig='/afs/cern.ch/user/x/xiezhen/w1/lumical/CMSSW_3_5_0_pre5/src/RecoLuminosity/LumiDB/site-local-config.xml'
        frontierparser=frontierconfigParser.frontierconfigParser()
        frontierparser.parse(sitelocalconfig)
        print 'full frontier string'
        print parser.fullfrontierStr(parser.schemaname(),frontierparser.parameterdict())
    f5='frontier://LumiPrep(otherparameter=value)/CMS_LUMI_DEV_OFFLINE'
    parser=connectstrParser(f5)
    parser.parse()
    print parser.protocol(),parser.service(),parser.schemaname(),parser.needsitelocalinfo()
    
    
