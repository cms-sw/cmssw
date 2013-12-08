from BeautifulSoup import *
import copy, os, sys

## PackageSplitter class is used for splitting package web page (pages.html) to increase usability
class PackageSplitter:
    ## Constructor method.
    # @param path is the reference manual directory path and it is used as destination and source.
    # @param prefix is used for file naming as prefix.
    def __init__(self, path, prefix = 'package_'):
        self.path           = path
        self.prefix         = prefix
        self.packageSource  = self.ReadFile('pages.html')
        self.WriteFile('pages_backup.html', self.packageSource)
        self.packages       = {}
        self.data           = None
            
    def ReadFile(self, fileName, pathFlag = True):
        """This method reads file directly or from path."""
        if pathFlag:
            print "Read:", self.path + fileName
            f = open(self.path + fileName)
        else:
            f = open(fileName)
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
    
    def __GetPackageName(self, str_):
        if str_.find('/') != -1:
            return str_[0:str_.find('/')]
        else:
            return str_
    
    def __GetFileName(self, str_):
        return self.prefix + str_.replace(' ', '_') + '.html'
    
    def GenerateTab(self, current = None, linkPrefix = ""):
        html = '<div class="tabs3">\n<ul class="tablist">\n'
        
        keys = self.data.keys()
        keys.sort()
        
        for i in keys:
            if i == current:
                html += '<li class="current"><a href="%s"><span>%s</span></a></li>\n' % (linkPrefix + self.__GetFileName(i), i)
            else:
                html += '<li><a href="%s"><span>%s</span></a></li>\n' % (linkPrefix + self.__GetFileName(i), i)
        
        html += '</ul>\n</div>'
        
        return html
    
    def CreateSubPage(self, packageName):
        if not self.data:
            self.PrepareData()
        tab = self.GenerateTab(current = packageName)
        counter = 0
        htmlList = '<table class="directory">\n<tbody>\n'
        keysI = self.data[packageName].keys()
        keysI.sort()
        for i in keysI:
            if counter % 2 == 0:
                htmlList += '<tr id="row_%d_" class="even">\n' % counter
            else:
                htmlList += '<tr id="row_%d_">\n' % counter
            htmlList += '<td class="entry">\n<img src="ftv2node.png" alt="o" width="16" height="22">\n'
            htmlList += '<a class="el" href="%s" target="_self">%s</a>\n' % (self.data[packageName][i], i)
            htmlList += '</td>\n<td class="desc">\n</td>\n</tr>\n'
            
            counter += 1
        htmlList += '</tbody>\n</table>\n'
        
        temp = copy.deepcopy(self.packageSource)
        soup = BeautifulSoup(temp)
        list_  = soup.find('div', { "class" : "directory" })
        list_.replaceWith(htmlList)
        
        tab_  = soup.find('ul', { "class" : "tablist" })
        tab_.replaceWith(tab_.prettify() + tab)
        
        data = str(soup.prettify())
        
        self.WriteFile(self.__GetFileName(packageName), data.replace('&lt;','<').replace('&gt;', '>'))
    
    def CreatePackagePage(self, outputFile):
        if not self.data:
            self.PrepareData()
        
        html = self.GenerateTab()
        
        temp = copy.deepcopy(self.packageSource)
        soup = BeautifulSoup(temp)
        tab  = soup.find('ul', { "class" : "tablist" })
        tab.replaceWith(tab.prettify() + html)
        
        data = str(soup.prettify())
        
        self.WriteFile(outputFile, data.replace('&lt;','<').replace('&gt;', '>'))
        
        for i in self.data:
            self.CreateSubPage(i)

        for i in self.data:
            for j in self.data[i]:
                self.AddTab2Page(self.data[i][j])

    def AddTab2Page(self, packageLink):
        tabHTML = self.GenerateTab(linkPrefix = "../../")

        orgSource = self.ReadFile(packageLink)

        soup = BeautifulSoup(orgSource)
        tab  = soup.find('ul', { "class" : "tablist" })
        tab.replaceWith(tab.prettify() + tabHTML)
        data = str(soup.prettify())
        self.WriteFile(packageLink, data.replace('&lt;','<').replace('&gt;', '>'))
    
    def PrepareData(self):
        soup        = BeautifulSoup(self.packageSource)
        contents    = soup.find("div", { "class" : "contents" })
        self.tr          = contents.findAll("tr", {})
        
        for i in self.tr:
            self.packages[i.text] = i.a["href"]
            
        self.data = {}
        
        for i in self.packages:
            if not "Package" in i:
                continue 
            if not self.data.has_key(self.__GetPackageName(i).replace(u'Package ', u'')):
                self.data[self.__GetPackageName(i).replace(u'Package ', u'')] = {}
            
            self.data[self.__GetPackageName(i).replace(u'Package ', u'')][i] = self.packages[i]
        

if len(sys.argv) == 4:
    PATH = sys.argv[1]
    OUTF = sys.argv[2]
    PREF = sys.argv[3]
    
    l = PackageSplitter(PATH, PREF)
    
    l.CreatePackagePage(OUTF)
else:
    print "parameter error. It must be like this:\nrun.py CMSSW/doc/html/ INPUT_FILE PREFIX\nExample: PackageSplitter.py OUTPUT_PATH/ pages.html packageDocumentation_"
