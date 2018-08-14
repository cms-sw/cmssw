from __future__ import print_function
import os,json,sys


STREAM    = "EXPRESS"
JSON_NAME = "certif.json"



def splitByTag(line,tags=["td","th"]):
  values = []
  pos = 0
  while pos > -1:
    firstTag=None
    firstTagPos=len(line)
    for tag in tags:
      posTag=line.find("<"+tag,pos)
      if posTag<firstTagPos and posTag>-1:
	firstTag=tag
	firstTagPos=posTag
    if not firstTag:
      break
    tag=firstTag
    posStartTag     = line.find("<"+tag,pos)
    posStartContent = line.find(">",posStartTag)+1
    posEnd 	    = line.find("</"+tag,posStartContent)
    pos = posEnd
    values.append(line[posStartContent:posEnd])
  return values

def getComment(line):
   tag="<span title=\""
   startComment=line.find(tag)+len(tag)
   stopComment=line.find("\"",startComment)
   return(line[startComment:stopComment])

def getRunQuality(fName=JSON_NAME):
  runQuality = {}
  if os.path.isfile(fName):
    with open(fName) as f:
      runQuality=json.load(f)
  return runQuality

def getHTMLtable(fName=".certif_temp.html"):
  table=""
  title =""
  with open(fName,"r") as certifFile:
    certifRaw = certifFile.read()
    if len(certifRaw)<100:
      return(0,0)
    title = certifRaw[certifRaw.find("<title>")+len("<title>"):certifRaw.find("</title>")]
    table=certifRaw.split("table>")[1]
  
  return(title,table)


def generateJSON():  
  # Getting certification HTML file
  os.system("wget http://vocms061.cern.ch/event_display/RunList/status.Collisions17.html -O .certif_temp.html > /dev/null 2>&1") 
  
  if not os.path.isfile(".certif_temp.html"):
    print("Unable to download file")
    return(1)
  
  runQuality = getRunQuality()
  
  if runQuality == {}:
    print("Warning, no %s found. Creating new list."%JSON_NAME)
  
  (title,table)=getHTMLtable()
  if table==0:
    print("Error, Unable to download run table.")
    return(1)
  runQuality["Last update"] = title[title.find("(")+1:title.find(")")]
  
  #Clean table and split per line
  table.replace("\n","").replace("<tr>","")
  table=table.split("</tr>")


  lenExpected = -1 	#Expected width of line
  colNumber = 0		#Column id to read status

  for line in table:
    entry = splitByTag(line)
    if lenExpected < 0 : 
      lenExpected = len(entry)
      for i in range(len(entry)):
        if STREAM.lower() in entry[i].lower():
          colNumber = i
    else:
      if len(entry)==lenExpected:
        comment=getComment(entry[colNumber])
        runQuality[entry[0]]={"qual":entry[colNumber][:4],"comment":comment}
      elif len(entry)>0:
        print("Error, unrecognized line !")
        return 1
    

  with open(JSON_NAME,'w') as data_file:    
      data_file.write(json.dumps(runQuality))


  os.system("rm .certif_temp.html")
  return(0)

def get():
   if generateJSON()!=0:
      print("ERROR, JSON file not updated... Loading old file.")
   
   if not JSON_NAME in os.listdir("."):
      print("ERROR, no JSON file available...")
      return({})
   else:
      return getRunQuality()

def checkRun(runNumber, runQuality):
   if not str(runNumber) in runQuality.keys():
      print("WARNING : no certification info for run %s"%runNumber)
      return(0)
   else:
      print("Data certification for run %s is %s"%(runNumber,runQuality[str(runNumber)]["qual"]))
      if not "GOOD" in runQuality[str(runNumber)]["qual"]:
         print("Reason : %s"%runQuality[str(runNumber)]["comment"])
      return("GOOD" in runQuality[str(runNumber)]["qual"])

if __name__ == "__main__":
   generateJSON()
   qual = get()
   for key in qual.keys():
      checkRun(int(key),qual)

