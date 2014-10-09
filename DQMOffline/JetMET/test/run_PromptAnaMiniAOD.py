##-----
# Set job-specific inputs based on shell
# the following enviromental variables
# export COSMIC_MODE=True # or False
# export JOB_NAME="CosmicStream112220"
# export NEVENTS=1000
# export ALL_HISTS=True
# export TRIGGER_SET=HLT
# export READ_LIST_FROM_FILE=False # or True
# export INPUTFILES='' # root file(s)
# export INPUTFILES_LIST='inputfile_BeamHaloExpress_120015.txt'
# for details see https://twiki.cern.ch/twiki/bin/view/CMS/JetMETDQMPromptAnalysis
##-----
import os
import FWCore.ParameterSet.Config as cms
#
# --- [cosmic sequence (default=True)?]
iscosmics = (os.environ.get('COSMIC_MODE','False'))
print 'iscosmics (default=True) = '+str(iscosmics)
#
# --- [name of job & output file (default=test)?]
jobname = (os.environ.get('JOB_NAME','test'))
print 'jobname (default=test) = '+str(jobname)
#
# --- [number of events (default=1000)]
nevents = int(os.environ.get('NEVENTS','1000'))
print 'nevents (default=1000) = '+str(nevents)
#
# --- [turn on all histograms (default=True)?]
allhist = (os.environ.get('ALL_HISTS','True'))
print 'allhist (default=True) = '+str(allhist)
#
#--- [read list of input files from a text file? or not (default=False)]
read_from_file = (os.environ.get('READ_LIST_FROM_FILE','False'))
print 'read list of input files from a text file (default=False) = '+str(read_from_file)
#
#--- [trigger set (default=HLT)]
trigger_set = (os.environ.get('TRIGGER_SET','HLT'))
print 'trigger set name (default=HLT) = '+str(trigger_set)
#
#--- [define list of input files]
inputfiles = []
if read_from_file=="True":
  #--- [name of the text file (default=inputfile_list_default.txt)]
  filename = (os.environ.get('INPUTFILES_LIST','inputfiles.txt'))
  file=open(filename)
  print file.read()
  f = open(filename)
  try:
    for line in f:
        inputfiles.append(line)
  finally:
    f.close()
else:
  inputfiles = os.environ.get('INPUTFILES',
'/store/relval/CMSSW_7_2_0_pre5/JetHT/MINIAOD/GR_R_72_V2_RelVal_jet2012D-v1/00000/682028D7-DE30-E411-A544-002618943874.root'
)
print 'List of input files'
print inputfiles
#-----

#
#-----
process = cms.Process("test")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

#
# DQM
#
#process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

#
# BeamHaloData producer
#
from Configuration.Geometry.GeometryIdeal_cff import *
### process.load("Configuration/StandardSequences/Geometry_cff") ### Deprecated
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
#process.load("RecoMET/Configuration/RecoMET_BeamHaloId_cff")
#process.GlobalTag.globaltag ='GR_R_38X_V13A::All'

#process.GlobalTag.globaltag ='GR_P_V14::All'
#process.GlobalTag.globaltag ='GR_R_52_V3::All'
process.GlobalTag.globaltag ='GR_70_V2_AN1::All'

# the task - JetMET objects
if iscosmics =="True":
  process.load("DQMOffline.JetMET.jetMETDQMOfflineSourceCosmic_cff")
else:
  process.load("DQMOffline.JetMET.jetMETDQMOfflineSourceMiniAOD_cff")

#change values for first jet and met analyzer parameterset -> all other parametersets are cloned from these
process.jetDQMAnalyzerAk4CaloUncleaned.OutputMEsInRootFile = cms.bool(True)
process.jetDQMAnalyzerAk4CaloUncleaned.OutputFileName = cms.string("jetMETMonitoring_%s.root" % jobname)
process.jetDQMAnalyzerAk4CaloUncleaned.TriggerResultsLabel = cms.InputTag("TriggerResults","",trigger_set)
process.jetDQMAnalyzerAk4CaloUncleaned.processname = cms.string(trigger_set)
#process.tcMetDQMAnalyzer.OutputMEsInRootFile = cms.bool(True)
#process.tcMetDQMAnalyzer.OutputFileName = cms.string("jetMETMonitoring_%s.root" % jobname)
#process.tcMetDQMAnalyzer.TriggerResultsLabel = cms.InputTag("TriggerResults","",trigger_set)
#process.tcMetDQMAnalyzer.processname = cms.string(trigger_set)
process.caloMetDQMAnalyzer.OutputMEsInRootFile = cms.bool(True)
process.caloMetDQMAnalyzer.OutputFileName = cms.string("jetMETMonitoring_%s.root" % jobname)
process.caloMetDQMAnalyzer.TriggerResultsLabel = cms.InputTag("TriggerResults","",trigger_set)
process.caloMetDQMAnalyzer.processname = cms.string(trigger_set)
#process.jetMETAnalyzer.TriggerResultsLabel = cms.InputTag("TriggerResults","","HLT8E29")
#process.jetMETAnalyzer.processname = cms.string("HLT8E29")

#if allhist=="True":
#  process.jetMETAnalyzer.DoJetPtAnalysis = cms.untracked.bool(False)
#  process.jetMETAnalyzer.DoJetPtCleaning = cms.untracked.bool(False)
#  process.jetMETAnalyzer.DoIterativeCone = cms.untracked.bool(False)

#process.jetMETAnalyzer.caloMETAnalysis.verbose = cms.int32(1)


################################################################################
#
# If allSelection = True, all the MET cleaning folders are filled.
# If allSelection = False, only All, BasicCleanup and ExtraCleanup are filled.
#
################################################################################
# if allhist=="True":
#   process.jetMETAnalyzer.caloMETAnalysis.allSelection       = cms.bool(True)
#   process.jetMETAnalyzer.caloMETNoHFAnalysis.allSelection   = cms.bool(True)
#   process.jetMETAnalyzer.caloMETHOAnalysis.allSelection     = cms.bool(True)
#   process.jetMETAnalyzer.caloMETNoHFHOAnalysis.allSelection = cms.bool(True)
#   process.jetMETAnalyzer.pfMETAnalysis.allSelection         = cms.bool(True)
#   process.jetMETAnalyzer.tcMETAnalysis.allSelection         = cms.bool(True)
#   process.jetMETAnalyzer.mucorrMETAnalysis.allSelection     = cms.bool(True)


# check # of bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

# for igprof
#process.IgProfService = cms.Service("IgProfService",
#  reportFirstEvent            = cms.untracked.int32(0),
#  reportEventInterval         = cms.untracked.int32(25),
#  reportToFileAtPostEvent     = cms.untracked.string("| gzip -c > igdqm.%I.gz")
#)

#
# /Wmunu/Summer09-MC_31X_V3-v1/GEN-SIM-RECO

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( ['/store/relval/CMSSW_7_2_0_pre5/JetHT/MINIAOD/GR_R_72_V2_RelVal_jet2012D-v1/00000/682028D7-DE30-E411-A544-002618943874.root'] );



#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring(inputfiles))

#
process.source.inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*')

#
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)
process.Timing = cms.Service("Timing")

## # Comment this out or reconfigure to see error messages 
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('caloMetDQMAnalyzer'),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        jetMETAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(1)
        ),
        noLineBreaks = cms.untracked.bool(True),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('DEBUG')
    ),
    categories = cms.untracked.vstring('jetMETAnalyzer'),
    destinations = cms.untracked.vstring('cout')
)


process.load('Configuration/StandardSequences/EDMtoMEAtJobEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')
#
cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/JetMET/'+str(cmssw_version)+'/Harvesting'
process.dqmSaver.workflow = Workflow


process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
    #outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string("reco_DQM_%s.root" % jobname)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True) ## default is false

)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

if iscosmics=="True":
  process.p = cms.Path(#process.BeamHaloId
#                       process.fixedGridRhoFastjetAllCalo
                    #process.dump
                    #process.jetMETDQMOfflineSourceCosmic
                    #* process.dqmStoreStats
###                     * process.MEtoEDMConverter
                     )
else:
  process.p = cms.Path(
                     process.jetMETDQMOfflineSource*
                     process.dqmStoreStats* process.dqmSaver
                     )

process.outpath = cms.EndPath(process.FEVT)
