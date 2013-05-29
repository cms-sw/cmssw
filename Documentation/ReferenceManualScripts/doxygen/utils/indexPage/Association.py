'''
Created on Aug 29, 2011

@author: MantYdze
'''

errorOnImport = False

import sys

try:
    import urllib2, os, json
except:
    errorOnImport = True    

cvsBaseUrl = "http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW"     # NO SLASH IN THE END
baseUrl = "http://cmssdt.cern.ch/SDT/doxygen/"
refmanfiles = {}
packageDocLinks = []


def parseJSON(url):
    ret = {}
    
    html = os.popen("curl \""+url+"\"")
    input = html.read()
    html.close()
    
    input = input.replace("{","").replace("}", "")
    collections = input.split("], ")
    
    for collection in collections:
        parts = collection.split(": [")
        title = parts[0].replace('"', '')
        list = parts[1].replace("]", "").replace('"', '').split(", ")
        ret[title] = list
    
    return ret


## Prepate dictionary of doxygen generated html files
def prepareRefManFiles(DOC_DIR):    
    output = os.popen("find "+DOC_DIR+" -wholename '*/class*.html' -not \( -name '*-members.html' \) -print")
    lines = output.read().split("\n")
    output.close()
    
    for line in lines:
        (head, tail) = os.path.split(line)
        refmanfiles[tail.replace("class","").replace(".html","")] = baseUrl+line[line.find(CMSSW_VERSION):]

## Extract links to package documentation
def preparePackageDocumentationLinks(DOC_DIR):
    source = open(DOC_DIR+"/pages.html", "r")
    lines = source.read().split("\n")
    source.close()
    
    for line in lines:
        if (line.find("li><a class=\"el\" href=\"") != -1):
            packageDocLinks.append(line.split("\"")[3])

## Format CVS link
def formatCVSLink(package, subpackage):
    cvsLink = "[ <a target=\"_blank\" href=\""+cvsBaseUrl+"/"+package+"/"+subpackage+"\">cvs</a> ]"
    return cvsLink

def formatPackageDocumentationLink(package, subpackage):    
    for link in packageDocLinks:
        if (link.find(package+"_"+subpackage+".html") != -1):
            return "[ <a target=\"_blank\" href=\"../"+link+"\">packageDoc</a> ]"
    
    return ""

## Fetches information about Subsystems/Packages/Subpackages from TagCollector
def generateTree(release):
    #data = json.loads(urllib2.urlopen('https://cmstags.cern.ch/tc/CategoriesPackagesJSON?release=' + release).read())
    data = parseJSON('http://mantydze.web.cern.ch/mantydze/tcproxy.php?type=packages&release=' + release)
    
    tree = {}
    subsystems = sorted(data.keys())
    
    for subsystem in subsystems:
        tree[subsystem] = {}
        for packagesub in data[subsystem]:        
            (package, subpackage) = packagesub.split("/")
            
            if not package in tree[subsystem]:
                tree[subsystem][package] = []
            tree[subsystem][package].append(subpackage)
            
    return (tree, subsystems)

## Generates HTML for subpackage
def generateLeavesHTML(SRC_DIR, package, subpackage):
    html = ""
    try:
        dirList=os.listdir(SRC_DIR + "/" + package+"/"+subpackage+"/interface/")
        for classfile in dirList:
            if classfile.endswith(".h"):
                classfile = classfile.replace(".h", "")
		for key in refmanfiles.keys():
                    if (key.find(classfile) != -1):
                        classfile = "<a target=\"_blank\" href=\""+refmanfiles[key]+"\">"+classfile+"</a>"
                        break
                
                html += "<li>"+classfile+"</li>"
    except:
        pass
    
    if html != "":
        html = "<ul>"+html+"</ul>"
    
    return html

## Generates HTML for Subsystem    
def generateBranchHTML(SRC_DIR, tree, branch): 
    branchHTML = ""
    
    for package,subpackages in sorted(tree[branch].items()):
        branchHTML += "<li><span><strong>"+package+"</strong></span><ul>"
        
        for subpackage in subpackages:
            branchHTML += "<li>"+subpackage + " "+ formatCVSLink(package, subpackage) + " " + formatPackageDocumentationLink(package, subpackage)
            branchHTML += generateLeavesHTML(SRC_DIR, package, subpackage)
            branchHTML+="</li>"
            
        branchHTML +="</ul>"
    
    branchHTML += "</li>"    
    return branchHTML

## Read template file
def loadTemplates():
    templateFile = SCRIPTS_LOCATION+"/indexPage/tree_template.html" 
            
    fileIN = open(templateFile, "r")
    treeTemplateHTML = fileIN.read()
    fileIN.close()
    
    
    templateFile = SCRIPTS_LOCATION+"/indexPage/indexpage_template.html"  
    fileIN = open(templateFile, "r")
    indexPageTemplateHTML = fileIN.read()
    fileIN.close()
    
    
    return treeTemplateHTML, indexPageTemplateHTML

if len(sys.argv) < 3:
    print "Not enough arguments, try script.py CMSSW_VERSION PROJECT_LOCATION SCRIPT_LOCATION"
    sys.exit()  

CMSSW_VERSION = sys.argv[1] 
PROJECT_LOCATION = sys.argv[2]                 
SCRIPTS_LOCATION = sys.argv[3]

SRC_DIR = PROJECT_LOCATION+"/src"    

try:
# Tree Preparation
    (treeTemplateHTML, indexPageTemplate) = loadTemplates()
    prepareRefManFiles(PROJECT_LOCATION+"/doc/html")
    preparePackageDocumentationLinks(PROJECT_LOCATION+"/doc/html")
    (tree, subsystems) = generateTree(CMSSW_VERSION)

    ## Index Page Preparations

    # Loading responsibles for subsystems
    #(managers, users) = json.loads(urllib2.urlopen('https://cmstags.cern.ch/tc/CategoriesManagersJSON').read())
    managers = parseJSON('http://mantydze.web.cern.ch/mantydze/tcproxy.php?type=managers')
    users = parseJSON('http://mantydze.web.cern.ch/mantydze/tcproxy.php?type=users')

except:
    ## Warning page
    fileIN = open(SCRIPTS_LOCATION+"/indexPage/indexpage_warning.html", "r")
    indexPageTemplateHTML = fileIN.read()
    fileIN.close()
    reason = "Failed during preparing treeview. "
    if errorOnImport:
        reason += " Error on import"
    output = open(PROJECT_LOCATION+"/doc/html/index.html", "w")
    output.write(indexPageTemplateHTML.replace("@CMSSW_VERSION@", CMSSW_VERSION).replace("@REASON@", reason))
    output.close()
    sys.exit(0) 


indexPageHTML = ""
indexPageRowCounter = 0
indexPageBlock = """
<tr class=\"@ROW_CLASS@\">
    <td width=\"50%\"><a href=\"#@SUBSYSTEM@\" onclick=\"javascript:getIframe('@SUBSYSTEM@')\">@SUBSYSTEM@</a></td>
    <td width=\"50%\" class=\"contact\">@CONTACTS@</td>
</tr>
<tr><td colspan=\"2\"><span id=\"@SUBSYSTEM@\"></span></td></tr>
"""

indexPageBlockNoTree = """
<tr class=\"@ROW_CLASS@\">
    <td width=\"50%\">@SUBSYSTEM@</td>
    <td width=\"50%\" class=\"contact\">@CONTACTS@</td>
</tr>
<tr><td colspan=\"2\"><span id=\"@SUBSYSTEM@\"></span></td></tr>
"""

# Links to Twiki pages
map = {}
map["Full Simulation"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSimulation"
map["Generators"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideEventGeneration"
map["Fast Simulation"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFastSimulation"
map["L1"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideL1Trigger"
map["HLT"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideHighLevelTrigger"
map["Reconstruction"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideReco"
map["Core"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrameWork"
map["DQM"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideDQM"
map["Database"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideCondDB"
map["Calibration and Alignment"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideCalAli"
map["Analysis"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideCrab"
map["Geometry"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideDetectorDescription"
map["DAQ"] = "https://twiki.cern.ch/twiki/bin/view/CMS/TriDASWikiHome"
map["Visualization"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideVisualization"
map["Documentation"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuide"



## Generating treeviews
for subsystem in subsystems:
    print subsystem
    branchHTML = generateBranchHTML(SRC_DIR, tree, subsystem)

    treeHTML = treeTemplateHTML.replace("@TREE@", branchHTML).replace("@SUBSYSTEM@", subsystem).replace("@CMSSW_VERSION@", CMSSW_VERSION)
            
    ## Formating index page's pieces
    block = indexPageBlockNoTree
    if (map.has_key(subsystem)):
        block = indexPageBlock
        treeHTML = treeHTML.replace("@LINK_TO_TWIKI@", map[subsystem])
        
    
    contacts = ""
    for manager in managers[subsystem]:
        if (contacts != ""):
            contacts += ", "
        contacts += "<a href=\"mailto:"+users[manager][1]+"\">" + users[manager][0] + "</a>" 

    
    if indexPageRowCounter % 2 == 0:
        rowCssClass = "odd"
    else:
        rowCssClass = "even"
    
    indexPageHTML += block.replace("@CONTACTS@", contacts).replace("@SUBSYSTEM@", subsystem).replace("@ROW_CLASS@", rowCssClass)

    output = open(PROJECT_LOCATION+"/doc/html/splittedTree/"+subsystem+".html", "w")
    output.write(treeHTML)
    output.close()

indexPageHTML = indexPageTemplate.replace("@TREE_BLOCKS@", indexPageHTML).replace("@CMSSW_VERSION@", CMSSW_VERSION)
output = open(PROJECT_LOCATION+"/doc/html/index.html", "w")
output.write(indexPageHTML)
output.close()    
