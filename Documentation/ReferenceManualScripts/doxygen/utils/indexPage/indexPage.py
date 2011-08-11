import sys

def changeVersion(CMSSW_Version):
    fh = open("index_template.php",'r')
    html = fh.read()
    fh.close()

    html = html.replace("CMSSW_Version",CMSSW_Version)

    fh = open("index.php",'w')
    fh.write(html)
    fh.close()
    
    print "Version "+CMSSW_Version+" added to index.php page"
    
if len(sys.argv) > 1:
    changeVersion(sys.argv[1])    
else:
    print "Not enough parameters: indexPage.py CMSSW_Version"
