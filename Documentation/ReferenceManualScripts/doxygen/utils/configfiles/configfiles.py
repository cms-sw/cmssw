import sys

def generateConfigFilesPage(PROJECT_LOCATION):
    input = open(PROJECT_LOCATION+"/doc/html/namespaces.html", "r")
    lines = input.read()
    input.close()
    
    # put nasty UL list to one line
    lines = lines.replace("\n<ul>", "<ul>")
    lines = lines.replace("<ul>\n", "<ul>")
    lines = lines.replace("\n</ul>", "</ul>")
    lines = lines.replace("</ul>\n", "</ul>")
    lines = lines.replace("\n<li>", "<li>")
    lines = lines.replace("<li>\n", "<li>")
    
    
    lines = lines.split("\n")
    html = []
    
    contentStarted = False  # table content started
    for line in lines:
        append = True;          # line is cff or cfi or cfg
        
        if (line.find("</table>") != -1) and (contentStarted):
            contentStarted = False
            
        if (contentStarted):
            if (line.find("<tr><td class=\"indexkey\">") == -1):    # Line stucture is wrong
                append = False;
            else:
                if (line.find("__cff.html") == -1) and \
                   (line.find("__cfg.html") == -1) and \
                   (line.find("__cfi.html") == -1):
                    append = False;
            
        if (line.find("<table>") != -1) and (not contentStarted):
            contentStarted = True            
            
        if (append):
            html.append(line)        
            
    # end "for line in lines"    
            
    html = "\n".join(html)
    
    output = open(PROJECT_LOCATION+"/doc/html/configfiles.html", "w")         
    output.write(html)
    output.close()
    
if len(sys.argv) > 1:
    PROJECT_LOCATION = sys.argv[1]
    
    generateConfigFilesPage(PROJECT_LOCATION)
    
    print "configfiles.py done"
else:
    print "Not enough parameters: configfiles.py PROJECT_LOCATION"    
    
    
