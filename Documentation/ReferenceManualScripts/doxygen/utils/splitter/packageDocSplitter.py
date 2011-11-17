'''
Created on Oct 27, 2011

@author: MantYdze
'''
import sys
from BeautifulSoup import BeautifulSoup, NavigableString

SUBSYSTEMS = {}
TAGLIST = {}

def parseTagList(tagList):
    input = open(tagList, "r")
    source = input.read()
    input.close()
    lines = source.split("\n")
    
    for line in lines[4:-3]:
        items = line.strip().split(" ")
        package = items[0]
        tag = items[-1]
        TAGLIST[package] = tag
        
def addTagToPackageDoc(line, subsystem, package):
    if (TAGLIST.has_key(subsystem+"/"+package)):
        tag = TAGLIST[subsystem+"/"+package]
    
        path = line[line.find("href=\"")+6:line.find("\">")]
        
        input = open(PROJECT_PATH+path, "r")
        source = input.read()
        input.close()
        
        output = open(PROJECT_PATH+path, "w")
        output.write(source.replace("@CVS_TAG@", tag).replace("(CVS tag: @)", "(CVS tag: "+tag+") "))
        output.close()

def extractList(filename):
    input = open(filename, "r")
    source = input.read()
    input.close()
    
    header = ""
    headerFull = False
    footer = ""
    footerStarted = False
    
    lines = source.split("\n")
    
    for line in lines:
                
        if (line.find("<li><a class=\"el\"") != -1) and (line.find("Package ") != -1):
            headerFull = True
            title = line[line.find("Package ")+8:-4]
            subsystem = title.split("/")[0]
            package = "/".join(title.split("/")[1:])
            if not SUBSYSTEMS.has_key(subsystem):
                SUBSYSTEMS[subsystem] = {}
            SUBSYSTEMS[subsystem][package] = line.replace("<li>","")
            
            addTagToPackageDoc(line, subsystem, package)
        
        if not headerFull:
            header += line+"\n"
        
        if line.find("<hr class=\"footer\"/>") != -1:
            footerStarted = True
        
        if footerStarted:
            footer += line+"\n"

    return header, footer, source


def addMenuToHeader(header, subsys):
    menu  = "<div class=\"tabs3\">\n"
    menu += "<ul class=\"tablist\">\n"
    
    for subsystem in sorted(SUBSYSTEMS.keys()):
        current = ""
        if subsystem == subsys:
            current = " class=\"current\""
        menu += "<li"+current+"><a href=\"packageDocumentation_"+subsystem+".html\"><span>"+subsystem+"</span></a></li>\n"
    
    menu += "</ul>\n"
    menu += "</div>\n"
    
    soap = BeautifulSoup(header)
    div = soap.find("div", {"class":"tabs"})
    
    div.append(NavigableString(menu))
    
    return soap.renderContents()
    
def createHTMLFiles(header, footer, PROJECT_PATH):
    for subsystem in sorted(SUBSYSTEMS.keys()):
        html = addMenuToHeader(header, subsystem)
        html += "<ul>"
        
        for package in sorted(SUBSYSTEMS[subsystem].keys()):
            html+="<li>"+SUBSYSTEMS[subsystem][package]+"</li>"
        html+="</ul>"
        
        output = open(PROJECT_PATH+"packageDocumentation_"+subsystem+".html", "w")
        output.write(html)
        output.close()
        
        
if len(sys.argv) > 2:
    filename = sys.argv[1]
    PROJECT_PATH = sys.argv[2]+"/doc/html/"
    tagList = sys.argv[2]+"/"+sys.argv[3]
    
    parseTagList(tagList)
    (header, footer, html) = extractList(PROJECT_PATH+filename)
    createHTMLFiles(header, footer, PROJECT_PATH)

    html = addMenuToHeader(html, "")
    output = open(PROJECT_PATH+filename, "w")
    output.write(html)
    output.close()








