
import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

#--- [read list of input files from a text file? or not (default=False)]
read_from_file = (os.environ.get('READ_LIST_FROM_FILE','False'))
print 'read list of input files from a text file (default=False) = '+str(read_from_file)
#
#
# --- [do harvesting (default=True)? or read in histogram files]
harvesting = (os.environ.get('HARVESTING',True))
print 'harvesting (default=True) = '+str(harvesting)
#
# --- [reference histogram (default=jetMETMonitoring_test.root)]
reference_histogram_file = (os.environ.get('REFERENCE_HISTOGRAM_FILE','jetMETMonitoring_test.root'))
print 'reference_histogram_file = '+str(reference_histogram_file)
#
# --- [input file(s) for harvesting/certification (default=reco_DQM_test.root)]
input_files = []
if read_from_file=="True":
  #--- [name of the text file (default=inputfile_list_default.txt)]
  filename = (os.environ.get('INPUTFILES_LIST','inputfile_list_default.txt'))
  file=open(filename)
  print file.read()
  f = open(filename)
  try:
    for line in f:
        input_files.append(line)
  finally:
    f.close()
else:
  input_root_files = os.environ.get('INPUTEDMFILES','file:reco_DQM_test.root').split(",")
  print 'input_root_files = '+str(input_root_files)
  print
  
  if harvesting:
    for file in input_root_files:        # Second Example
      input_files.append(str(file))
    else:
      test_histogram_file = os.environ.get('TEST_HISTOGRAM_FILE','jetMETMonitoring_test.root')
      print 'test_histogram_file = '+str(test_histogram_file)
print
    


#-----------------------------
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

#-----------------------------
# DQM Environment & Specify inputs
#-----------------------------
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1)
)

#
#--- When read in RECO file including EDM from ME
process.source = cms.Source("PoolSource",
    processingMode = cms.untracked.string('RunsAndLumis'),
    fileNames = cms.untracked.vstring(input_files)
)

#-----

process.load('Configuration/StandardSequences/EDMtoMEAtRunEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')
#
cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/JetMET/'+str(cmssw_version)+'/Harvesting'
process.dqmSaver.workflow = Workflow

#-----------------------------
# Specify root file including reference histograms
#-----------------------------
process.DQMStore.referenceFileName = reference_histogram_file

#-----------------------------
# Locate a directory in DQMStore
#-----------------------------
process.dqmInfoJetMET = cms.EDAnalyzer("DQMEventInfo",
                subSystemFolder = cms.untracked.string('JetMET')
                )

#-----------------------------
# JetMET Certification Module 
#-----------------------------
process.load("DQMOffline.JetMET.dataCertificationJetMET_cff")

if harvesting:
  print
else:
  process.dataCertificationJetMET.fileName = cms.untracked.string(test_histogram_file)

#-----------------------------
# 
#-----------------------------
process.load("DQMOffline.Trigger.JetMETHLTOfflineClient_cfi")
from DQMOffline.Trigger.JetMETHLTOfflineClient_cfi import *

#-----------------------------
# 
#-----------------------------
#process.p = cms.Path(process.dqmInfoJetMET*process.dataCertificationJetMET)

process.p = cms.Path(process.EDMtoME
                     * process.dqmInfoJetMET
                     * process.jetMETHLTOfflineClient
                     * process.dataCertificationJetMETSequence
                     * process.dqmSaver)

