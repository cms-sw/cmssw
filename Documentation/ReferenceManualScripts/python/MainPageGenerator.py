import json, urllib2, os, sys
from BeautifulSoup import *

## MainPageGenerator class is used for generating main page that contains domain trees (Analysis, Calibration and Alignment, Core, DAQ etc.) 
class MainPageGenerator:
    ## Constructor method.
    # @param dataPath parameter gives path of data directory that contains .js, .css and image files needed for generating tree pages
    # @param path is the reference manual directory path and it is used as destination and source.
    # @param cmsVer is version of CMSSW.
    def __init__(self, dataPath, path, cmsVer = ""):
        self.path = path
        self.dataPath = dataPath

        self.CMSVER             = cmsVer

        self.managersURL        = 'http://cmsdoxy.web.cern.ch/cmsdoxy/tcproxy.php?type=managers'
        self.usersURL           = 'http://cmsdoxy.web.cern.ch/cmsdoxy/tcproxy.php?type=users'
        self.CMSSWURL           = 'http://cmsdoxy.web.cern.ch/cmsdoxy/tcproxy.php?type=packages&release=CMSSW_4_4_2'
        
        self.tWikiLinks         = {'Analysis':'https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideCrab',
                                   'Calibration and Alignment':'https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideCalAli',
                                   'Core':'https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrameWork',
                                   'DAQ':'https://twiki.cern.ch/twiki/bin/view/CMS/TriDASWikiHome',
                                   'DQM':'https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideDQM',
                                   'Database':'https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideCondDB',
                                   'Documentation':'https://twiki.cern.ch/twiki/bin/view/CMS/SWGuide',
                                   'Fast Simulation':'https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFastSimulation',
                                   'Full Simulation':'https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSimulation',
                                   'Generators':'https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideEventGeneration',
                                   'Geometry':'https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideDetectorDescription',
                                   'HLT':'https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideHighLevelTrigger',
                                   'L1':'https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideL1Trigger',
                                   'Reconstruction':'https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideReco',
                                   'Visualization':'https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideVisualization'}
        
        self.data               = None
        
        self.GitLink            = "https://github.com/cms-sw/cmssw/tree/" + self.CMSVER + "/%s/%s"
        
        self.title = "<center>\n<h1>CMSSW Documentation</h1>\n<h2>" + self.CMSVER + "</h2>\n</center>\n"
        self.links = """
<p style="margin-left:10px;">
Learn <a href="ReferenceManual.html">how to build Reference Manual</a><br>
Learn more about <a target="_blank" href="http://www.stack.nl/~dimitri/doxygen/commands.html">special doxygen commands</a>
</p>\n\n"""
        self.head = """
<!-- Content Script & Style -->
<script type="text/javascript">
var itemList = [];

function toggleHoba(item, path)
{
    for(var i = 0; i < itemList.length; i++)
    {
        if(itemList[i] == item)
        {
            var iframe = $("#"+itemList[i]+"_div").children("iframe:first");
            if(!iframe.attr("src"))
            {
                iframe.attr("src", path)
            }
            $("#"+item+"_div").slideToggle();
        }
        else
            $("#"+itemList[i]+"_div").slideUp();
    }
}

$(document).ready(function() {
searchBox.OnSelectItem(0);
$(".doctable").find("td").each(function(){ if (this.id.indexOf("hoba_") != -1)itemList.push(this.id);});
});
</script>
<style>
.DCRow
{
    background: #eeeeff;
    border-spacing: 0px;
    padding: 0px;
    border-bottom: 1px solid #c1c1dc;
}

.DCRow:hover
{
    background: #cde4ec;
}
</style>
<!-- Content Script & Style -->
        """
        self.contentStamp       = '$CONTENT$'
        self.mainPageTemplate   = self.ReadFile("index.html")
        self.WriteFile("index_backup.html", self.mainPageTemplate)          #backup file
        soup     = BeautifulSoup(self.mainPageTemplate)
        soup.head.insert(len(soup.head), self.head)
        
        contents = soup.find("div", { "class" : "contents" })
        for child in contents.findChildren():
            child.extract()
        contents.insert(0, self.contentStamp)
        self.mainPageTemplate = str(soup)
        self.mainPageTemplate = self.mainPageTemplate.replace("CSCDQM Framework Guide", "")
        self.mainPageTemplate = self.mainPageTemplate.replace('&lt;','<').replace('&gt;', '>')
        print "Main page template created..."

        self.CreateBuildRefMan()
        print "RefMan created..."
        
        self.treePageTamplate   = self.ReadFile(self.dataPath + "tree_template.html", pathFlag = False)
        self.classesSource      = self.ReadFile("classes.html")
        self.filesSource        = self.ReadFile("files.html")
        self.packageSource      = self.ReadFile("pages.html")
        
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
        
    def GetFileName(self, fileName):
        """This method returns file name without extension"""
        if '.' in fileName:
            return fileName[0:fileName.find('.')]
        else:
            return fileName
    
    def ParseJsonFromURL(self, URL):
        """This method returns data which is read from URL"""
        u = urllib2.urlopen(URL)
        return json.loads(u.read())
    
    def __ParseItem(self, str_):
        return str_[0:str_.find('/')]
    
    def __ParseSubItem(self, str_):
        if '/' in str_:
            return str_[str_.find('/')+1:]
        else:
            return None
        
    def __GetHTMLItemDepth(self, item):
        return item["id"].count("_") - 1 # 1 for doxygen 1.8.5, 2 for old ver.
    
    def __HTMLFileName(self, fileName):
        return fileName.lower().replace(' ', '_')
    
    def PrepareData(self):
        self.managers = self.ParseJsonFromURL(self.managersURL)
        print "Managers loaded and parsed..."
            
        self.users = self.ParseJsonFromURL(self.usersURL)
        print "Users loaded and parsed..."
        
        self.data = {}
        for i in self.managers.keys():
            self.data[i] = {"__DATA__":{"Contact":[]}}
            for j in self.managers[i]:
                self.data[i]["__DATA__"]["Contact"].append(self.users[j])
        self.domains = self.ParseJsonFromURL(self.CMSSWURL)
        print "Domains loaded and parsed..."
        
        for i in self.domains.keys():
            for j in self.domains[i]:
                if self.__ParseItem(j) not in self.data[i]:
                    self.data[i][self.__ParseItem(j)] = {}
                if self.__ParseSubItem(j) not in self.data[i][self.__ParseItem(j)]:
                    self.data[i][self.__ParseItem(j)][self.__ParseSubItem(j)] = {}
                
                self.data[i][self.__ParseItem(j)][self.__ParseSubItem(j)]["__DATA__"] = {
                    'git': self.GitLink % (self.__ParseItem(j), self.__ParseSubItem(j))
                    }
                
        # for getting package links
        soup        = BeautifulSoup(self.packageSource)
        contents    = soup.find("div", { "class" : "contents" })
        li          = contents.findAll("tr", {})
        
        self.packages = {}
        for i in li:
            if i.a["href"]:
                self.packages[i.a.text] = i.a["href"]
        print "Packages parsed(%d)..." % len(self.packages)

        # for getting items from file.html
        soup        = BeautifulSoup(self.filesSource)
        contents    = soup.find("div", { "class" : "contents" })
        tr          = contents.findAll("tr", {})
        self.classes= {}
        origin = 0 
        if tr[0].text == 'src': origin = -1
        # depth of interface items can be only 3
        flag = False
        for i in tr:
            if self.__GetHTMLItemDepth(i) + origin == 1:
                self.classes[i.text]    = {}
                level1          = i.text
                flag = False
                
            if self.__GetHTMLItemDepth(i) + origin == 2:
                self.classes[level1][i.text] = {}
                level2 = i.text
                flag = False

            if self.__GetHTMLItemDepth(i) + origin == 3 and i.text == u'interface':
                flag = True
            if self.__GetHTMLItemDepth(i) + origin == 3 and i.text != u'interface':
                flag = False
                
#            print i.text, self.__GetHTMLItemDepth(i)
#            raw_input()
            
            if flag and i.text != u'interface':
                self.classes[level1][level2][i.text] = i.a["href"]
                #self.ZEG = i
        print "Class hierarchy loaded(%d)..." % len(self.classes)
        
#        self.WriteFile("dbg.json", json.dumps(self.classes, indent = 1))
        
        # for parsing classes links from classes.html
        soup        = BeautifulSoup(self.classesSource)
        contents    = soup.find("div", { "class" : "contents" })
        td          = contents.findAll("td", {})
        self.classesURLs = {}
        # add items to self.classesURLs
        for i in td:
            if i.a and 'href' in i.a:
                self.classesURLs[i.a.text] = i.a['href']
        print "Class URLs was loaded... (%s)" % len(self.classesURLs)
        
        for i in self.data.keys():
            for j in self.data[i].keys():
                if j not in self.classes: continue
                for k in self.data[i][j].keys():
                    if "Package " + j + "/" + k in self.packages:
                        self.data[i][j][k]["__DATA__"]["packageDoc"] = '../' + self.packages["Package " + j + "/" + k]
                    if k not in self.classes[j]: continue
                    for h in self.classes[j][k]:
                        if self.GetFileName(h) in self.classesURLs:
                            self.data[i][j][k][self.GetFileName(h)] = {"__DATA__": '../' + self.classesURLs[self.GetFileName(h)]}
                        else:
                            self.data[i][j][k][self.GetFileName(h) + ".h"] = {"__DATA__": '../' + self.classes[j][k][h]}

    def ExportJSON(self, fileName):
        if self.data == None:
            self.PrepareData()
        self.WriteFile(fileName, json.dumps(self.data, indent = 1))
    
    def CreateBuildRefMan(self):
        content = """<h1>The Reference Manual </h1>
        This is the CMSSW Reference Manual, the reference documentation of all classes and packages in CMSSW.<p>
        This page explains how to write the documentation for your code.

        </p><h2>Class Documentation</h2>

        Classes and methods are documented with properly formatted <a target="_blank" class="el" href="d3/d88/namespacecomments.html">comments</a> in the code.<p>
        Here is a template of a documented <a target="_blank" href="http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/Documentation/CodingRules/Template.h?rev=HEAD&amp;cvsroot=CMSSW&amp;content-type=text/vnd.viewcvs-markup">.h file</a>, and of a <a target="_blank" href="http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/Documentation/CodingRules/Template.cc?rev=HEAD&amp;cvsroot=CMSSW&amp;content-type=text/vnd.viewcvs-markup">.cc file</a>. The resulting doxygen page is <a target="_blank" class="el" href="d6/d3e/classTemplate.html">here</a>.

        </p><h2>Package Documentation</h2>

        Each package should contain a very brief description of its content and purpose. Remember that this is a reference, and not a user's guide: tutorials, howtos, etc. are best documented in the <a target="_blank" href="https://twiki.cern.ch/twiki/bin/view/CMS/SWGuide">CMS Offline Guide</a> and in the <a target="_blank" href="https://twiki.cern.ch/twiki/bin/view/CMS/WorkBook">WorkBook</a>. Cross links between the CMS Offline Guide and the WorkBook and this manual are a good way to avoid duplication of content.<p>
        This documentation should be written in a file [Package]/doc/[Package].doc. The simplest way of doing this is to go to the doc/ directory in your package and then run the script  
        <a target="_blank" href="http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/*checkout*/CMSSW/Documentation/ReferenceManualScripts/scripts/makePackageDoc?rev=HEAD&amp;cvsroot=CMSSW">makePackageDoc</a>,
         which is available in your PATH.

        </p><h2> How to generate your documentation locally </h2>
        One you have updated your documentation, you can look at how it displays in the following way:

         <ul>
           <li>check out the following packages:  
        <pre> &gt; cmsrel CMSSW_7_X_X
         &gt; cd CMSSW_7_X_X/
         &gt; cmsenv
         &gt; git cms-addpkg Documentation

         &gt; generate_reference_manual

         wait...

         &gt; firefox doc/html/index.html </pre>
          </li>
        </ul>"""
        self.WriteFile('ReferenceManual.html', self.mainPageTemplate.replace(self.contentStamp, content))
        
    def CreateNewMainPage(self, outputFileName):
        if self.data == None:
            self.PrepareData()
            
        contents = """
        <table class="doctable" border="0" cellpadding="0" cellspacing="0">
        <tbody>
        <tr class="top" valign="top">
        <th class="domain">Domain</th><th class="contact">Contact</th>
        </tr>
        """
        keysI = sorted(self.data.keys())
        for i in keysI:
            #########################
            if i == 'Other': continue
            
            self.__NewTreePage(i)
            contents = contents + '\n<tr class="DCRow">\n'    ######### TAG: TR1
            #########################
            if i == 'Operations':
                contents = contents + """<td width="50%%" style="padding:8px">%s</td>\n""" % i
            else:
                contents = contents + """<td width="50%%" style="padding:8px;cursor:pointer" onclick="toggleHoba('hoba_%s', 'iframes/%s.html')" id="hoba_%s"><a>%s</a></td>\n""" % (i.replace(' ', '_'), i.lower().replace(' ', '_'), i.replace(' ', '_'), i)
            #########################
            
            contents = contents + '<td width="50%" class="contact">'
            for j in range(len(self.data[i]["__DATA__"]["Contact"])):
                if j == len(self.data[i]["__DATA__"]["Contact"]) - 1:
                    contents = contents + '<a href="mailto:%s">%s</a> ' % (self.data[i]["__DATA__"]["Contact"][j][1], self.data[i]["__DATA__"]["Contact"][j][0])
                else:
                    contents = contents + '<a href="mailto:%s">%s</a>, ' % (self.data[i]["__DATA__"]["Contact"][j][1], self.data[i]["__DATA__"]["Contact"][j][0])
            contents = contents + '</td>\n'
            contents = contents + '</tr>\n\n'               ######### TAG: TR1
            #########################
            if i == 'Operations': continue
            #########################
            contents = contents + """
            <tr><td colspan="2" style="background:#d7dbe3">
            <div style="display:none;" id="hoba_%s_div"><iframe width="100%%" frameborder="0"></iframe></div>
            </td></tr>
            """ % (i.replace(' ', '_'))
            
        contents = contents + "</table>"
        self.WriteFile(outputFileName, self.mainPageTemplate.replace(self.contentStamp, self.title + contents + self.links))
    
    def __NewTreePage(self, domain):
        
        if domain not in self.data: return
        
        content = ''
        keysI = sorted(self.data[domain].keys())
        for i in keysI:
            if i == '__DATA__': continue
            content += self.HTMLTreeBegin(i)
            keysJ = sorted(self.data[domain][i].keys())
            for j in keysJ:
#                if len(self.data[domain][i][j].keys()) == 1:
#                    if self.data[domain][i][j].has_key("__DATA__"):
#                        content += self.HTMLTreeAddItem(j, self.data[domain][i][j]["__DATA__"])
#                    else:
#                        content += self.HTMLTreeAddItem(j)
#                    continue
                keysK = sorted(self.data[domain][i][j].keys())
                length = len(keysK)
#                content += "<!-- Begin -->"
                if length > 1:
                    if "__DATA__" in self.data[domain][i][j]:
                        content += self.HTMLTreeBegin(j, self.data[domain][i][j]["__DATA__"])
                    else:
                        content += self.HTMLTreeBegin(j)
                else:
                    if "__DATA__" in self.data[domain][i][j]:
                        content += self.HTMLTreeAddItem(j, self.data[domain][i][j]["__DATA__"], folder = True)
                    else:
                        content += self.HTMLTreeAddItem(j, folder = True)
                
                for k in keysK:
                    if k == '__DATA__': continue
                    if self.data[domain][i][j][k]["__DATA__"]: content += self.HTMLTreeAddItem(k, self.data[domain][i][j][k]["__DATA__"])
                    else: content += self.HTMLTreeAddItem(k)
                if length > 1:
                    content += self.HTMLTreeEnd()
#                content += "<!-- End -->"
            content += self.HTMLTreeEnd()
        if domain in self.tWikiLinks:
            self.WriteFile("iframes/%s.html" % domain.lower().replace(' ', '_'), self.treePageTamplate % (domain, self.tWikiLinks[domain], content))
        else:
            print 'Warning: The twiki link of "%s" domain not found...' % domain
            self.WriteFile("iframes/%s.html" % domain.lower().replace(' ', '_'), self.treePageTamplate % (domain, '#', content))
    
    def HTMLTreeBegin(self, title, links = {}):
        html = '\n<li>\n<div class="hitarea expandable-hitarea"></div>\n'
        html = html + '<span class="folder">%s\n' % title
        for i in links.keys():
            html = html + '<a target="_blank" href="%s">[%s]</a> \n' % (links[i], i)
        html = html + '</span>\n'
        html = html + '<ul style="display: block;">\n'
        return html
    
    def HTMLTreeEnd(self):
        return '</li></ul>\n\n'
    
    def HTMLTreeAddItem(self, title, links = None, endNode = False, folder = False):
        if endNode: html = '\t<li class="last">'
        else: html = '\t<li>'
        
        if isinstance(links, str) or isinstance(links, type(u'')):
            if folder:
                html = html + '\t<a href="%s" target="_blank" class=""><span class="emptyFolder">%s</span></a>\n' % (links, title)
            else:
                html = html + '\t<a href="%s" target="_blank" class=""><span class="file">%s</span></a>\n' % (links, title)
        elif isinstance(links, dict):
            if folder:
                html = html + '<span class="emptyFolder">%s ' % title
            else:
                html = html + '<span class="file">%s ' % title
            for i in links.keys():
                html = html + '<a target="_blank" href="%s">[%s]</a> \n' % (links[i], i)
            html = html + '</span>'
        else:
            html = html + '\t<span class="file">%s</span>\n' % title
        return html + '\t</li>\n'
        
if len(sys.argv) == 5:
    DATA_PATH = sys.argv[1]
    PATH = sys.argv[2]
    VER  = sys.argv[3]
    OUTF = sys.argv[4]
      
    #os.system("cp -rf %s../data/iframes/ %s" % (os.path.split(__file__)[0], PATH))
    
    l = MainPageGenerator(DATA_PATH, PATH, cmsVer = VER)
    
    l.CreateNewMainPage(OUTF)
else:
    print "parameter error. It must be like this: python MainPageGenerator.py DATA_PATH/ CMSSW/doc/html/ CMS_VER OUTPUT_FILE_NAME"
