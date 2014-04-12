from BeautifulSoup import *
import copy, sys

## ConfigFiles class is used for generating 'config files' html page.
class ConfigFiles:
    ## Constructor method.
    # @param path is the reference manual directory path and it is used as destination and source.
    # @param outputFile is output file name. It should be "configfiles.html" for backward compatibility.
    # @param prefix is used for file naming as prefix.
    def __init__(self, path, outputFile, prefix = "configfilesList_"):
        self.path           = path
        self.outputFile     = outputFile
        self.prefix         = prefix
        self.keywords       = ["_cff", "_cfi", "_cfg"]
        self.NSSource       = self.ReadFile('namespaces.html')
        self.tableStamp     = '$TABLE$'
        self.navBarStamp    = '$NAVBAR$'
        self.Template       = None
        
        self.data           = {}
    
    def ReadFile(self, fileName, pathFlag = True):
        """This method reads file directly or from path."""
        if pathFlag:
            print "Read:", self.path + fileName
            f = open(self.path + fileName)
        else:
            f = open(os.path.split(__file__)[0] + '/' + fileName)
            print "Read:", fileName
        data = f.read()
        f.close()
        return data
    
    def WriteFile(self, fileName, data):
        """This method writes data"""
        print "Write:", self.path + fileName
        f = open(self.path + fileName, "w")
        f.write(data)
        f.close()
    
    def CreateTemplate(self):
        self.Template = copy.deepcopy(self.NSSource)
        soup     = BeautifulSoup(self.Template)
        contents = soup.find("div", { "class" : "contents" })
        navBar   = soup.find("div", { "id" : "navrow2", "class" : "tabs2"})
        for child in contents.findChildren():
            child.extract()
        contents.insert(0, self.tableStamp)
        navBar.insert(len(navBar), self.navBarStamp)
        self.Template = str(soup)
        print "Template created..."
    
    def Check(self, str_):
        for i in self.keywords:
            if i in str_:
                return True
        return False
    
    def PrepareData(self):
        
        soup     = BeautifulSoup(self.NSSource)
        contents = soup.find("div", { "class" : "contents" })
        td       = contents.findAll("td", {})
        
        parsedITemCounter = 0
        for i in td:
            if i.a and self.Check(i.a.text):
                if not self.data.has_key(i.a.text[0].upper()):
                    self.data[i.a.text[0].upper()] = {}
                self.data[i.a.text[0].upper()][i.a.text] = i.a["href"]
                parsedITemCounter += 1
        print "Config files parsed(%d)..." % parsedITemCounter
    
    def CreateTab(self, current):
        tabHTML = '<div class="tabs3">\n<ul class="tablist">\n'
        keys = self.data.keys()
        keys.sort()
        for i in keys:
            if current == i:
                tabHTML += '<li class="current"><a href="%s"><span>%s</span></a></li>\n' % ('#', i)
            else:
                tabHTML += '<li><a href="%s"><span>%s</span></a></li>\n' % (self.FileName(i), i)
        tabHTML += '</ul>\n</div>\n'
        return tabHTML
    
    def CreateTable(self, current):
        tableHTML = '<table width="100%">\n<tbody>\n'
        for i in self.data[current].keys():
            tableHTML += '<tr><td class="indexkey"><a class="el" href="%s"> %s </a></td><td class="indexvalue"></td></tr>' % (self.data[current][i], i)
        tableHTML += '</tbody>\n</table>\n'
        return tableHTML
    
    def FileName(self, current):
        return self.prefix + current + ".html"
        
    def CreateConfigFilePage(self):
        self.PrepareData()
        self.CreateTemplate()
        
        keys = self.data.keys()
        keys.sort()
        
        for i in keys:
            page = self.Template.replace(self.navBarStamp, self.CreateTab(i)).replace(self.tableStamp, self.CreateTable(i))
            self.WriteFile(self.FileName(i), page)
            if i == 'A': self.WriteFile('configfiles.html', page)
        

if len(sys.argv) == 3:
    PATH = sys.argv[1]
    OUTF = sys.argv[2]
    
    l = ConfigFiles(PATH, OUTF)
    
    l.CreateConfigFilePage()
else:
    print "parameter error. It must be like this: python ConfigFiles.py CMSSW/doc/html/ output.html"
