import sys
import os
import re
from BeautifulSoup import BeautifulSoup

BASE = "http://cmssdt.cern.ch/SDT/doxygen/"
INDEX = {}
printOutput = False;

def replace(regex,replacement,content):
    p = re.compile(regex,re.IGNORECASE);
    c = p.sub(replacement,content)
    return c 

def findMatchingFiles(w, source_htmls):
    ret = ""
    for srcFile in source_htmls:
        if srcFile.split("/")[-1].__str__().find(w) != -1:
            ret +=  " " + srcFile
            
    return ret

def filter(s,w,k):
    o = s.split()
    if len(o) > 1:
        betterChoice = ""
        for i in range(len(o)):
            if re.search("[^a-zA-Z]"+w+"[^a-zA-Z]", o[i]):
                if re.search(".*"+k+".*",o[i]):
                    return o[i]
                else:
                    if betterChoice == "" or len(betterChoice) > o[i]:
                        betterChoice = o[i]
        return betterChoice
    else:
        if re.search("[^a-zA-Z]"+w+"[^a-zA-Z]", s):
            return s
        else:
            return ""
        
def getLink(word):
        
    if word.isdigit() or (len(word) < 5):
        return ""
    
    out = filter(findMatchingFiles(word, py_source_htmls),word,"")
    if not out or out == "":
        out = filter(findMatchingFiles(word, h_source_htmls),word,"")
        if not out or out == "":
            return ""
    return BASE+out.lstrip()

def process(filename):
    
    if (filename != None) and (len(filename) < 5):
        return
    
    fh = open(filename,'r')
    html = fh.read()
    fh.close()

  
    content = ""
    # find only code block
    soup = BeautifulSoup(html)
    pres = soup.findAll("pre", {"class":"fragment"})
    
    for pre in pres:
        if pre.contents != None:
            content += pre.renderContents()
    # END OF find only code block

    # remove links
    content = replace(r'<a\b[^>]*>(.*?)</a>','',content)
    
    content = content.replace("&#39;", "'")    
    content = content.replace("&quot;", '"')
    
    matches = []
    tmp = re.findall('[\w,\.]+_cf[i,g,f]',content)
    for t in tmp:
        matches.extend(t.split("."))
        
    matches.extend(re.findall('"\w+"',content))
    matches.extend(re.findall("'\w+'",content))
    
    set = {}                                  # 
    map(set.__setitem__, matches, [])         # removing duplicate keywords
    matches = set.keys()                      # 
    
    for match in matches:
        
        match = match.replace("'", "")    
        match = match.replace('"', "")
        
        if (INDEX.has_key(match)):
            href = INDEX[match]
        else:
            href = getLink(match)
        
        if (href != ""):
            INDEX[match] = BASE+href[href.find("CMSSW_"):]
            
            link = "<a class=\"configfileLink\" href=\""+href+"\">"+match+"</a>"
            regex = r"\b"+match+r"\b"
            html = replace(regex, link, html)
            
            ########################
            if printOutput:
                print ">>>>>["+match+"]",
            ########################
        
            ########################
            if printOutput:
                print href
            ########################
        
    fh = open(filename,'w')
    fh.write(html)
    fh.close()

if len(sys.argv) > 1:

    DIR = sys.argv[1] +"/doc/html/"
              
    global py_source_htmls
    global h_source_htmls

    h_source_htmls = []
    py_source_htmls = []
    
    print "ieskau h_source"
    
    query = "find "+DIR+" -name '*8h_source.html' -print"
    output = os.popen(query)
    h_source_htmls = output.read().split("\n")
    
    print "ieskau py_source"
    
    query = "find "+DIR+" -name '*8py_source.html' -print"
    output = os.popen(query)
    py_source_htmls = output.read().split("\n")
   
    query = 'find '+DIR+' \( -name "*cf[i,g,f]*py*html" -or -name "namespace*cf[i,g,f].html" \) -print '
    output = os.popen(query)
    files = output.read().split("\n")
    i = 0
    for file in files:
        i = i + 1
        print i.__str__()+") "+file
        process(file)
    print "-----------------------------------------------------------"    
else:
    print "not enough parameters"
