import xml.dom.minidom
class cacheconfigParser(object):
    def __init__(self):
        self.__parameterDict={}

    def parseXMLfile(self,filename):
        f=open(filename,'r')
        parseXMLstr(f.read())

    def parseXMLstr(self,document):
        dom.xml.dom.minidom.parseString(document)
        dom.getElementsByTagName('frontier-connect')
        
    def parameterDict(self):
        return self.__parameterDict
    
if __name__ == '__main__':
    pass
