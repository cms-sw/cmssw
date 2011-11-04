'''
Created on Oct 27, 2011

@author: MantYdze
'''
import sys
from BeautifulSoup import BeautifulSoup, NavigableString

SUBSYSTEMS = {}

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
    
    (header, footer, html) = extractList(PROJECT_PATH+filename)
    createHTMLFiles(header, footer, PROJECT_PATH)

    html = addMenuToHeader(html, "")
    output = open(PROJECT_PATH+filename, "w")
    output.write(html)
    output.close()








