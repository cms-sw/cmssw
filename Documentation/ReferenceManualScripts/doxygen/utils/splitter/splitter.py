'''
Created on Nov 9, 2010
Updated on Oct 19, 2011

@author: Mantas Stankevicius
'''

import sys
from BeautifulSoup import BeautifulSoup, NavigableString

INDEX = {}
# Just dictionary of letters found, used in creation of menu

LINKS = []
# LINKS = List of lists of links [["A", "link_A_1", "link_A_2"],["B", "link_B_1"], ["C", "link_C_1"]]
# Structure of lists : 
# first element is letter
# others elements are <tr><td>LINK</td><td>WHATEVER</td></tr>

def getHeader():
    """ Reading source file until <table>. 
        After <table> begins list of links 
    """
    fh = open(sourceFile,'r')
    source = fh.read()
    fh.close()
    
    lines = source.split("\n")

    html = []
    enough = False
    
    for line in lines:
        if line.find("<table>") != -1:
            enough = True
            
        if (not enough):
            html.append(line)
            
    html.append("<table width=\"100%\">")
    return "\n".join(html)

def getFooter():
    """ Reading source file from end until </table>. 
        After </table> begins list of links (reading from end) 
    """
    fh = open(sourceFile,'r')
    source = fh.read()
    fh.close()
    
    lines = source.split("\n")
    lines.reverse()
    
    html = []
    enough = False
    
    for line in lines:
        if (not enough):
            html.append(line)
            
        if line.find("</table>") != -1:
            enough = True
        
    html.reverse()            
    return "\n".join(html)   

def extractLinks():
    """ Extracts links from source file 
        from <div class = 'contents'> </div>"""
        
    fh = open(sourceFile,'r')
    source = fh.read()
    fh.close()
    
    soup = BeautifulSoup(source)
    div = soup.find("div", {"class":"contents"})
    
    if (div != None):
        content = div.renderContents()
    
    lines = content.split("\n")
    for line in lines:
        if (line.find("<tr>") != -1):
            
            indexFrom = line.rfind(".html\">") + 7
            indexTo = line.rfind("</a>")
            linkText = line[indexFrom:indexTo]
            
            linkTextParts = linkText.split("::")
            
            if len(linkTextParts) == 2:
                tmpLine = line.replace(linkText, linkTextParts[1])
                letter = linkTextParts[1][0].upper()
                appendLinkToList(tmpLine, letter)

            letter = linkText[0].upper()
            appendLinkToList(line, letter)
            

def appendLinkToList(line, letter):
    if (not INDEX.has_key(letter)):
        subList = [letter, line]
        LINKS.append(subList)
        INDEX[letter] = letter
    else:
        for l in LINKS:
            if l[0] == letter:
                l.append(line)
    

def createMenu(letter):
    html  = "<div class=\"tabs3\">\n"
    html += "<ul class=\"tablist\">\n"

    letters = []
    for i in INDEX:
        letters.append(i)
    
    letters.sort()
    letters.append("All")
    
    for l in letters:
        c = l
        current = ""
        if c == letter:
            current = " class=\"current\""
            
        html += "<li"+current+"><a href=\""+PREFIX+c+".html\"><span>"+c+"</span></a></li>\n"
    
    html += "</ul>\n"
    html += "</div>\n"
    
    return html

def createHtmlPages():
    
    HTMLHeader = getHeader()
    HTMLFooter = getFooter()
    
    for list in LINKS:
        letter = list[0]
        
        html = HTMLHeader
        
        for item in list[1:]:
            html += item+"\n"
        
        html += HTMLFooter
        
        soap = BeautifulSoup(html)
        div = soap.find("div", {"class":"tabs2"})

        text = NavigableString(createMenu(letter))
        div.append(text)

#        div.insert(div.__len__(), createMenu(letter))
        
        html = soap.renderContents()
        
        path = PROJECT_LOCATION+"/doc/html/"+PREFIX+letter+".html"
        output = open(path, "w")
        output.write(html)
        output.close()
        
        if letter == "A":
            output = open(sourceFile, "w")
            output.write(html)
            output.close()  
        
        print PROJECT_LOCATION+"/doc/html/"+PREFIX+letter+".html    Done!"                  

def backupOriginal():
    fh = open(sourceFile,'r')
    html = fh.read()
    fh.close()

        
    soap = BeautifulSoup(html)
    div = soap.find("div", {"class":"tabs2"})
    # Adding menu of letters at the end of navigation bar
    text = NavigableString(createMenu("All"))
    div.append(text)
#    div.insert(div.__len__(), createMenu("All"))
    
    html = soap.renderContents()
    
    output = open(PROJECT_LOCATION+"/doc/html/"+PREFIX+"All.html", "w")
    output.write(html)
    output.close() 

if len(sys.argv) > 3:
    global PROJECT_LOCATION
    PROJECT_LOCATION = sys.argv[1]
    global sourceFile   
    sourceFile = PROJECT_LOCATION+sys.argv[2]
    global PREFIX
    PREFIX = sys.argv[3]
    
    # DO NOT CHANGE ORDER
    print "Reading source file"
    extractLinks()  
    backupOriginal()
    print "Creating html files"    
    createHtmlPages()
    
else:
    print "Not enough parameters: file.py PROJECT_LOCATION SOURCE_FILE PREFIX"
    print "example: file.py /data/CMSSW_4_3_0 /doc/html/namespaces.html namespaceList_"
    print "---"
    print "PROJECT_LOCATION - (i.e. /data/CMSSW_4_3_0)"
    print "---"
    print "SOURCE_FILE - /doc/html/namespaces.html"    
    print "SOURCE_FILE - /doc/html/configfiles.html"
    print "SOURCE_FILE - /doc/html/annotated.html"
    print "---"
    print "html file prefix when splitting source file by alphabetical order."
    print "Examples:"
    print "PREFIX - namespaceList_ (result: namespaceList_A.html, namespaceList_B.html,...)"
    print "PREFIX - configfilesList_ (result: configfilesList_A.html, configfilesList_B.html,...)"
    print "PREFIX - classesList_ (result: classesList_A.html, classesList_B.html,...)"
    print "---"
