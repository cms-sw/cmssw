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
print 'nevents (default=1000)  = '+str(nevents)
#
# --- [turn on all histograms (default=True)?]
allhist = (os.environ.get('ALL_HISTS','True'))
print 'allhist (default=True) = '+str(allhist)
#
#--- [read list of input files from a text file? or not (default=False)]
read_from_file = (os.environ.get('READ_LIST_FROM_FILE','True'))
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
  filename = (os.environ.get('INPUTFILES_LIST','inputfile_list_default.txt'))
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
  '/store/hidata/HIRun2010/HIAllPhysics/RECO/PromptReco-v2/000/150/314/D28FA8E9-4EEA-DF11-9B2A-000423D987E0.root').split(",")
  #'/store/data/Commissioning09/MinimumBias/RECO/v4/000/102/347/F85D1BC6-A06A-DE11-BDF8-0019B9F581C9.root').split(",")
  #'/store/data/CRAFT09/Calo/RECO/v1/000/112/220/F0B768A4-5E93-DE11-B222-000423D94524.root').split(",")

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
process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

###for HI running
#process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('RecoJets.JetAssociationProducers.trackExtrapolator_cfi')
process.load('RecoJets.JetAssociationProducers.ak7JTA_cff') 
process.load('RecoJets.JetAssociationProducers.ak5JTA_cff')
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag ='GR_P_V27A::All'

# the task - JetMET objects
if iscosmics =="True":
  process.load("DQMOffline.JetMET.jetMETDQMOfflineSourceCosmic_cff")
else:
  process.load("DQMOffline.JetMET.jetMETDQMOfflineSourceHI_cff")

process.jetMETAnalyzer.OutputMEsInRootFile = cms.bool(True)
process.jetMETAnalyzer.OutputFileName = cms.string("jetMETMonitoring_%s.root" % jobname)
process.jetMETAnalyzer.TriggerResultsLabel = cms.InputTag("TriggerResults","",trigger_set)
process.jetMETAnalyzer.processname = cms.string(trigger_set)

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
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(*inputfiles))

#
process.source.inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*')

#
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( nevents )
)
process.Timing = cms.Service("Timing")

## # Comment this out or reconfigure to see error messages 
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('jetMETAnalyzer'),
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

if iscosmics=="True":
  process.p = cms.Path(process.jetMETDQMOfflineSourceCosmic
                     * process.dqmStoreStats
                     * process.MEtoEDMConverter
                     )
else:
  process.p = cms.Path(process.jetMETDQMOfflineSource
                     * process.dqmStoreStats
                     * process.MEtoEDMConverter
                     )

process.outpath = cms.EndPath(process.FEVT)
process.DQM.collectorHost = ''


