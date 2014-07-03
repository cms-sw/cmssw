#!/usr/bin/env python
import os, sys, re
import xml.parsers.expat
from ROOT import *



# DQM Report Summary Parser ---- Yuri Gotra May-30-2008
# getDQMSummary.py ~/3d/DQM_R000043434.root



usage = "usage: %s <DQM_File.root>" %         os.path.basename(sys.argv[0])

if len(sys.argv) < 2:
   print usage
   sys.exit(2)
else:
   argv = sys.argv
   infile = argv[1]
   outfile = argv[2]
   print argv[1]
   f = TFile(infile, 'read')
   ff = open(outfile,'w')


reportSummary = "<reportSummary>"
DQMDataDir = "DQMData"
EventInfoDir = "/Run summary/EventInfo"
ReportSummaryContentsDir = "/Run summary/EventInfo/reportSummaryContents"
summary = std.vector(float)()
reportSummaryContents = std.vector(string)()
f.cd(DQMDataDir)
dirtmp = gDirectory
dirlist = dirtmp.GetListOfKeys()
iter = dirlist.MakeIterator()
key = iter.Next()
td = None

while key:
   td = key.ReadObj()
   dirName = td.GetName()
   if(re.search('Run (\d+)', dirName)):
       rundirname = dirName
       break
   key = iter.Next()
   
SummaryContentsDirExists = 0

def getDQMSegSummaryResult(f, subdet):
  global SummaryContentsDirExists
  SummaryContentsDirExists = 0
  ReportSummaryContentsDir = "/Run summary/EventInfo/reportSummaryContents"
  SummaryContentsDir = DQMDataDir + '/' + rundirname + '/' + subdet + ReportSummaryContentsDir
  SegEventInfoDir = DQMDataDir + '/' + rundirname + '/' + subdet + EventInfoDir
  f.cd(SegEventInfoDir)

  dirtmp = gDirectory
  dirlist = dirtmp.GetListOfKeys()
  iter = dirlist.MakeIterator()
  key = iter.Next()
  td = None

  while key:
     td = key.ReadObj()
     dirName = td.GetName()
     if(re.search('reportSummaryContents', dirName)):
        SummaryContentsDirExists = 1
        break
     key = iter.Next()
  if(SummaryContentsDirExists):
     f.cd(SummaryContentsDir)
  else:
     SummaryContentsDirExists = 0
     print "Warning: No reportSummaryContents directory found in", subdet
     return reportSummaryContents, summary
  SummaryContentsDir = gDirectory
  dirlist = SummaryContentsDir.GetListOfKeys()
  iter = dirlist.MakeIterator()
  key = iter.Next()
  tk = None
  while key:
      tk = key.ReadObj()
      keyName = tk.GetName()
      mn = re.search('(<.+?>)', keyName)
      if(mn):
             reportSummaryContent = mn.group(0)
             ms = re.split('=', keyName)
             m = re.search('-?\d?\.?\d+', ms[1])
             segsummary = m.group(0)
             summary.push_back(float(segsummary))
             reportSummaryContents.push_back(reportSummaryContent)
      key = iter.Next()
  return reportSummaryContents, summary
         
def getDQMDetSummaryResult():
    
   reportSummaryDir = DQMDataDir + '/' + rundirname + '/' + subdet + EventInfoDir

   f.cd(reportSummaryDir)
   reportSummaryDir = gDirectory
   dirlist = reportSummaryDir.GetListOfKeys()
   iter = dirlist.MakeIterator()
   key = iter.Next()
   tk = None
   while key:
      tk = key.ReadObj()
      keyName = tk.GetName()
      if(re.search(reportSummary, keyName)):
         ms = re.split('=', keyName)
         m = re.search('-?\d?\.?\d+', ms[1])
         detsummary = m.group(0)
         break
      key = iter.Next()
   return detsummary

DQMDataDir = "DQMData"

f.cd(DQMDataDir)
dirtmp = gDirectory
dirlist = dirtmp.GetListOfKeys()
iter = dirlist.MakeIterator()
key = iter.Next()
td = None

# controlling THE SUBDETECTOR dir list
while key:
 td = key.ReadObj()
 dirName = td.GetName()
 if(re.search('Run (\d+)', dirName)):
     rundirname = dirName
     break
 key = iter.Next()

RunDirFull = DQMDataDir + '/' + rundirname
f.cd(RunDirFull)
dirtmp = gDirectory
dirlist = dirtmp.GetListOfKeys()
iter = dirlist.MakeIterator()
key = iter.Next()


SubDetectors =std.vector(string)()

while key:
 td = key.ReadObj()
 dirName = td.GetName()
 print dirName
 SubDetectors.push_back(dirName)
 key = iter.Next()

print SubDetectors 

for subdet in SubDetectors:
   print >>ff, '============================'
   print >>ff, rundirname, subdet
   getDQMSegSummaryResult(f, subdet)
   if( SummaryContentsDirExists == 1):
      summary.push_back(float(getDQMDetSummaryResult()))
      reportSummaryContents.push_back(reportSummary)
   j = 0
   for i in summary:
       print >> ff, reportSummaryContents[j], int(1000*i)
       j = j + 1
   summary.clear()
   reportSummaryContents.clear()
print >> ff, '============================' 
