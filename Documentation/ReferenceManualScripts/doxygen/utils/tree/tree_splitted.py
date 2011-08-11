import sys
import os
import re

aFileList = [] 
fileList = [] # all generated files

def getEmptyAlfabeticalList():
    letters = ["A", "a", "B", "b", "C", "c", "D", "d", "E", "e", "F", "f", "G", "g", "H", "h", "I", "i", "J", "j", "K", "k", "L", "l", "M", "m", "N", "n", "O", "o", "P", "p", "Q", "q", "R", "r", "S", "s", "T", "t", "U", "u", "V", "v", "W", "w", "X", "x", "Z", "z", "_"]
    
    for letter in letters:
        sublist = [letter]
        aFileList.append(sublist)
    return aFileList

def make4thLevelLinks(packagePath):
    #classPath = "/build/ms"            
    classPath = os.popen("echo $PWD").read().strip()     # path to CMSSW build
    print classPath
    global counter4
    global CMSSW_Version                                                                
    global doxygenBaseUrl                                                               
    packagePath = classPath +"/"+CMSSW_Version+"/src/"+packagePath+"/interface/"
    print packagePath
    try:
        html = "<ul>"
        dirList=os.listdir(packagePath)                         # taking directory content from interface folder
        for className in dirList:
            dclassName = className
            if className.endswith(".h") and os.path.isfile(packagePath+className):           # be sure that item is file
                className = className.replace(".h", "")         # leaving only class name
                link = "<li>"+dclassName+"</li>"                 # default value in case no html file will be found to make a link
                fileLetter = className[0]
                if len(className) > 1:    
                    global aFileList
                    found = False
    
                    pattern = re.compile(".*/class.*"+className+".html$")
                    for letterList in aFileList:                    
                        if letterList[0] == fileLetter:
                            for file in letterList:
                                m = pattern.match(file)
                                if (m != None):
                                    found = True  
                                    link = "<li><a target=\"_blank\" href=\""+file+"\">"+className+"</a></li>"
                                    counter4 += 1
                                    
                            if (found == False):
                                global fileList
                                m = pattern.match(file)
                                for file in fileList:
                                    m = pattern.match(file)
                                    if (m != None):
                                        found = True  
                                        link = "<li><a target=\"_blank\" href=\""+file+"\">"+className+"</a></li>"
                                        counter4 += 1
                    
                html += link
        html += "</ul>"
        print html
    except:
        print "WARNING: make4thLevelLinks failed to find out where the class source files are stored!"
        html = ""
            
    return html

def makeFileList():         # list of generated files of doxygen    
    print "Searching class*.html files"
    output = os.popen("find "+fileBasePath+" -not \( -name '*members.html' \) -name 'class*.html' -print")
    raw = output.read()
    output.close()
    
    raw = raw.replace(fileBasePath+"/", "")
    global fileList
    fileList = raw.split("\n")
    
    global aFileList
    aFileList = getEmptyAlfabeticalList()    
    print "Preprocessing list of files"             # making 2 dimentional array: one dimention - letters, second - classnames beginning with that letter 
    for f in fileList:
        if (f.find("class")) != -1:
            fileLetter = f[f.find("class")+5]       # class first letter
            for letterList in aFileList:            
                if fileLetter == letterList[0]:
                    letterList.append(f)
    
def getLineElements(line):          # for creation tree skeleton (2, 3 levels)
    line = line.replace("\n", "")
    line = line.replace("*", "")  
     
    lineItems = line.split("\t")
    return lineItems

def getElement(line):               # returns first not empty element from line
    lineElements = getLineElements(line)
        
    for i in lineElements:
        element = i.replace("\t", "")
        element = element.strip()
        
        if len(element) != 0:
            return element
    return ""  

def getLineIdentifiers(line):               # return line identifiers ex. sim reco calib database
    lineElements = getLineElements(line)
    
    lineToReturn = ""
    for i in lineElements:
        element = i.replace("\t", "")
        element = element.strip()
        
        if len(element) != 0:
            lineToReturn = element
            
    return lineToReturn.split("/")  

def formatCVSLink(package, subPackage):
    global cvsBaseUrl
    
    cvsLink = "["+"<a target=\"_blank\" href=\""+cvsBaseUrl+"/"+package+"/"+subPackage+"\">cvs</a>]"
    return cvsLink

def formatDOCLink(package, subPackage):
    global doxygenBaseUrl
    
    doxygenLink = ""                        #default value in case no html will be found
    path = package+"_"+subPackage+".html"
    file = fileBasePath+"/"+path            
    if os.path.isfile(file):
        doxygenLink = "[<a target=\"_blank\" href=\""+path+"\">doc</a>]"
        
    return doxygenLink

def getListByName(fileName, identifier):    # returns list of branches and leafs
    fileIN = open(fileName, "r")
    line = fileIN.readline()
    allowFillList = False
    newBlockHeader = ""
    subList = []
    list = []    
    
    while line:    
        element = getElement(line)
        if line.find("*") != -1:                        # new block
            allowFillList = False
            newBlockHeader = element
        
        identifiersMatch = False    
        if identifier.lower() in getLineIdentifiers(line):
            identifiersMatch = True
            
        if (allowFillList == False) and (identifiersMatch):    # block we need    
            allowFillList = True
            subList = []
                
            if newBlockHeader != element:
                subList.append(newBlockHeader)
                                        
            list.append(subList)
        
        if (allowFillList == True) and (identifiersMatch):        
            subList.append(element)
                    
        line = fileIN.readline()
    fileIN.close()
    return list
        
# end of getListByName       
            
def getHtmlFromList(fileName, identifier, CMSSW_Version): 
    
    
    list = getListByName(fileName, identifier)   
    
    html = ""
    first = True
    firstElement = ""
    for subList in list:
        for currentElement in subList:
            if (subList.index(currentElement) == 0) and (first):
                if first == True:               # First in list 
                    first = False
                    firstElement = currentElement               
                html+="<li><span><strong>"+firstElement+"</strong></span><ul>"      # 2nd level
                
            else:
                packagePath = firstElement+"/"+currentElement
                cvsLink = formatCVSLink(firstElement, currentElement)
                doxygenLink = formatDOCLink(firstElement, currentElement)
                # creating 4th Layer
                global doxygenBaseUrl
                html += "<li>"+currentElement+" "+cvsLink+" "+doxygenLink        # 3rd level
                global counter3
                counter3 += 1
#		print packagePath
                html += make4thLevelLinks(packagePath)                              # 4th level
                html += "</li>"
                # end of creating 4th layer
                
        first = True                            # new branch
        html += "</ul>"
        
    html += "</li>"
    return html
    
# end of getHtmlFromList

def getTemplateHtml(fileName):
    
    return html

if len(sys.argv) < 3:
    print "Not enough arguments, try script.py destination data prefix1..."
    sys.exit()
destinationFile = "tree.html"
dataFile = sys.argv[1]                      # assosiation file                 

global CMSSW_Version
CMSSW_Version = sys.argv[2]                 

global fileBasePath
fileBasePath = CMSSW_Version+"/doc/html" # NO SLASH IN THE END

global doxygenBaseUrl
#doxygenBaseUrl = "http://cms-service-sdtweb.web.cern.ch/cms-service-sdtweb/doxygen/"+CMSSW_Version+"/doc/html/" 
doxygenBaseUrl = "http://cmssdt.cern.ch/SDT/doxygen/"+CMSSW_Version+"/doc/html/"            
# MUST BE SLASH IN THE END

global cvsBaseUrl
cvsBaseUrl = "http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW"     # NO SLASH IN THE END

html = ""
makeFileList()     

global counter3
counter3 = 0
global counter4
counter4 = 0

# TEMPLATE
templateFile = "utils/tree/data/template_splitted.html"                       
textToReplace = "TREE_HTML_GOES_HERE"                       #text in template which will be changed to html of tree
fileIN = open(templateFile, "r")
templateHtml = fileIN.read()
# END OF TEMPLATE

# links to Twiki pages
map = {}
map["sim"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSimulation"
map["gen"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideEventGeneration"
map["fastsim"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFastSimulation"
map["l1"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideL1Trigger"
map["hlt"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideHighLevelTrigger"
map["reco"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideReco"
map["core"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrameWork"
map["dqm"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideDQM"
map["db"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideCondDB"
map["calib"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideCalAli"
map["analysis"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideCrab"
map["geometry"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideDetectorDescription"
map["daq"] = "https://twiki.cern.ch/twiki/bin/view/CMS/TriDASWikiHome"
map["visualization"] = "https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideVisualization"


for argument in sys.argv[3:]:
    print argument
    treeHtml = getHtmlFromList(dataFile, argument, CMSSW_Version)

    treeHtml = templateHtml.replace(textToReplace, treeHtml)
    treeHtml = treeHtml.replace("</head>", "<base href=\""+doxygenBaseUrl+"\"/></head>")

	# adding links to Twiki
    	
    try:
        link = " | <a target=\"_blank\" href=\""+map[argument]+"\" style=\"text-decoration:none;\">\
	<img border=\"1\" alt=\"Twiki\" src=\"http://cmssdt.cern.ch/SDT/doxygen/common/twiki.gif\"/></a>"
    except:
	link = ""
        pass
    treeHtml = treeHtml.replace("LINK_TO_TWIKI", link)
	# END of adding links to Twiki
    
    
    output = open(CMSSW_Version+"/doc/html/splittedTree/"+argument+".html", "w")
    output.write(treeHtml)
    output.close()
    
    
    
print "3rd level links "+counter3.__str__()
print "4th level links "+counter4.__str__()
