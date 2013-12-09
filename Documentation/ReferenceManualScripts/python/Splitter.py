from BeautifulSoup import *
import sys

## Splitter class is used for splitting class and namespace web pages to reduce access time.
class Splitter:
    # @param path is the reference manual directory path and it is used as destination and source.
    # @param fileName is output file name. (You should consider the menu.js script (https://cmssdt.cern.ch/SDT/doxygen/common/menu.js) while you are naming output file)
    # @param prefix is used for file naming as prefix.
    def __init__(self, path, fileName, prefix):
        self.fileName = fileName
        self.filePath = path
        self.prefix   = prefix
        
        self.backupPrefix= prefix + 'backup_'
        self.headerStamp = '<div class="header">'
        self.contentStamp= '<div class="contents">'
        self.footerStamp = '<hr class="footer"/>'
        self.namespaceImg= 'ftv2ns.png'
        self.classImg    = 'ftv2cl.png'
        
        #Min depth
        self.minD        = 2
        self.chr         = '_'
        self.List        = ()
        
        file = open(path + fileName, 'r')
        self.source = file.read()
        file.close()
        
        self.headerPos   = self.source.find(self.headerStamp)
        self.contentPos  = self.source.find(self.contentStamp)
        self.footerPos   = self.source.find(self.footerStamp)

        self.parsedSource= None
        
    def GetHeader(self):
        return self.source[0:self.headerPos]

    def GetDivHeader(self):
        return self.source[self.headerPos:self.contentPos]
    
    def GetFooter(self):
        """This method returns footer of the input page"""
        return self.source[self.footerPos:len(self.source)]

    def GetContent(self):
        """This method returns content of the input page"""
        return self.source[self.contentPos:self.footerPos]
        
    def WritePage(self, fileName, data):
        f = open(self.filePath + fileName,  "w")
        f.write(data)
        f.close()
        
    def Backup(self):
        #backup
        self.WritePage(self.backupPrefix + 'deep_' + self.fileName, self.source)

        #editted backup
        edb = self.GetHeader() + self.CreateTab("All") + self.GetDivHeader() + self.GetContent() + self.GetFooter()
        self.WritePage(self.prefix + '_ALL.html', edb)
        
    def __GetName(self, node):
        if node.a:
            return node.a.contents[0]
        elif node.b:
            return node.b.contents[0]
        else:
            return "---------"
    
    def __GetInfo(self,  node):
        if node.findAll("td", {"class":"desc"}):
            return node.findAll("td", {"class":"desc"})[0].text
        else:
            return ""
    
    def __IsLink(self,  node):
        """This method returns true if it has link."""
        if(node.a): return True
        else: return False
        
    def __GetLink(self, node):
        """This method returns node link."""
        return node.a['href']
        
    def __GetDepth(self, node):
        """This method returns depth of the node. To determine this, '_' character is used."""
        # <tr id="row_0_" ... It has two '_' character
        # <tr id="row_0_0_" ... It has three '_' character
        return node["id"].count(self.chr) - self.minD

    def CreateTab(self, current):
        # If source code was not parsed
        if not self.parsedSource:
            self.parsedSource = self.GetNsC()
        header = '<div class="tabs3" align="center">\n<ul class="tablist">\n'
        if current == "All":
            all_ = '<li class="current"><a href="%s"><span>All</span></a></li>\n' % (self.prefix + '_ALL.html')
        else:
            all_ = '<li><a href="%s"><span>All</span></a></li>\n' % (self.prefix + '_ALL.html')
        footer = '</ul>\n</div>\n'
        tab    = ''
        
        tab    = header
        for i in self.List:
            if i != current:
                tab = tab + u'<li><a href="%s"><span>%s</span></a></li>\n' % ("%s%s.html" % (self.prefix, i),  i)
            else:
                tab = tab + u'<li class="current"><a href="%s"><span>%s</span></a></li>\n' % ("%s%s.html" % (self.prefix, current),  i)
            
        tab = tab + all_ + footer
        
        return tab.encode("ascii")
        
    def CreatePage(self, current, itemList):
        """This method creates web page."""
        data = ""
        data = self.GetHeader()
        data = data + self.CreateTab(current)
        data = data + self.GetDivHeader()
        data = data + '<div class="contents"><table width="100%">'
        
        for i in itemList:
            data = data + '<tr><td class="indexkey"><a class="el" href="%s">%s</a></td><td class="indexvalue">%s</td></tr>\n' % i
        
        data = data + '</table></div>'
        data = data + self.GetFooter()
        return data
    
    def CreatePages(self):
        self.Backup()
        self.CreateFirstPage('A')
        if not self.parsedSource:
            self.parsedSource = self.GetNsC()
		
        for i in self.parsedSource.keys():
            print i, "is ok..."
            self.WritePage("%s%s.html" % (self.prefix, i), self.CreatePage(i, self.parsedSource[i]))

    def CreateFirstPage(self, letter):
        if not self.parsedSource:
            self.parsedSource = self.GetNsC()
        self.WritePage(self.fileName,
                       self.CreatePage(letter, self.parsedSource[letter]))

    def GetNsC(self):
        content = self.GetContent()
        bs      = BeautifulSoup(content)
        
        tr      = bs.findAll("tr", {})
        data    = {}
        path    = {}
        
        #Structure of data variable:
        # {letter:[(link, ::name, info), ...]}
        #Example:
        # {'G':[
        #        (grEng.html,         'GraphicEngine', ''),
        #        (grEngTerra.html,    'GraphicEngine::TerrainGenarator', ''),
        #        (grEngTerraGVC.html, 'GraphicEngine::TerrainGenarator::GetVerticesCount', ''),
        # ],
        #  'H':[...]
        # }
        
        #Structure of path variable:
        # {depth:node}
        #Example:
        # {0:'GraphicEngine',
        #  1:'GraphicEngine::TerrainGenarator',
        #  2:'GraphicEngine::TerrainGenarator::GetVerticesCount'}
        for i in tr:
            if self.__GetDepth(i) == 0:
                path = {0:self.__GetName(i)}
            else:
                path[self.__GetDepth(i)] = path[self.__GetDepth(i) - 1] + "::" + self.__GetName(i)
                
            if self.__IsLink(i):
                if not path[self.__GetDepth(i)][0].upper() in self.List:
                    self.List = self.List + (path[self.__GetDepth(i)][0].upper(), )
                    data[path[self.__GetDepth(i)][0].upper()] = []
                    
                if not self.__GetName(i).upper()[0] in self.List:
                    data[self.__GetName(i)[0].upper()] = []
                
                if path[self.__GetDepth(i)] != self.__GetName(i):
                    data[path[self.__GetDepth(i)][0].upper()].append((self.__GetLink(i),
                                                                     path[self.__GetDepth(i)], 
                                                                     self.__GetInfo(i)))
                data[self.__GetName(i)[0].upper()].append((self.__GetLink(i),
                                                            self.__GetName(i), 
                                                            self.__GetInfo(i)))
        
        return data

if len(sys.argv) > 3:
    path = sys.argv[1]
    file = sys.argv[2]
    prefix = sys.argv[3]
    s = Splitter(path, file, prefix)
    print "pages are creating..."
    s.CreatePages()
else:
    print "Not enough parameters: file.py PATH FILE PREFIX"
    print "Example: python Splitter.py CMSSW/doc/html/ annotated.html annotated_"
