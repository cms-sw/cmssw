#!/usr/bin/env python
import os, sys, commands, re

#    globaltag = cms.string('FT_R_42_V10A::All'))
GTRX = "(.*)globaltag.*'(.*)'(.*)"
FILERX = "\s*fileNames\s*=\s*cms\.untracked\.vstring.*"
#FILENAME = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root"
FILENAME = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MultiJet__RAW__v1__180252__E85462BF-1A03-E111-84FA-BCAEC5364C42.root"
#relbase = os.environ.get('CMSSW_RELEASE_BASE', None)
base = os.environ.get('CMSSW_BASE')
CFGI = 'testPromptReco.py'
CFGO = 'testPromptReco2.py'
SNIPPET = """
process.IgProfService = cms.Service("IgProfService",
             reportFirstEvent            = cms.untracked.int32(1),
             reportEventInterval         = cms.untracked.int32(25),
             reportToFileAtPostEvent     = cms.untracked.string("| gzip -c > IgProf.%I.gz")
             )

process.load("DQMServices.Components.DQMStoreStats_cfi")
process.stats = cms.Path(process.dqmStoreStats)
process.schedule.insert(-2,process.stats)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))
"""
if os.path.exists(CFGI):
  os.remove(CFGI)
if os.path.exists(CFGO):
  os.remove(CFGO)

if base:
  print 'Running test 15 for release %s' % base
  commands.getoutput('eval `scram r -sh` && addpkg Configuration/DataProcessing')
  commands.getoutput('python %s/src/Configuration/DataProcessing/test/pp_reco_t.py' % base)
  if os.path.exists(CFGI):
    commands.getoutput('eval `scram r -sh` && addpkg Configuration/AlCa')
    execfile('%s/src/Configuration/AlCa/python/autoCond.py' % base)
    gt = autoCond.get('com10', None)
    if gt:
      print 'Using uptodate GT %s' % gt
      fi = open(CFGI, 'r')
      fo = open(CFGO, 'w')
      for line in fi:
        m = re.match(GTRX, line)
        if m:
          #print m.group(2)
          fo.write("%sglobaltag = cms.string('%s'%s" % (m.group(1), gt, m.group(3)))
          continue
        m = re.match(FILERX, line)
        if m:
          fo.write("fileNames = cms.untracked.vstring('file:%s')" % FILENAME)
          continue
        fo.write(line)
      fo.write(SNIPPET)
      fi.close()
      fo.close()
      os.rename(CFGO,CFGI)
else:
  print 'Error, no suitable release configured. Quitting'
  sys.exit(1)

