# Imports
import FWCore.ParameterSet.Config as cms

# Create a new CMS process
process = cms.Process('cluTest')

# Import all the necessary files
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023tiltedReco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# Number of events (-1 = all)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input file
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('root://eoscms//eos/cms/store/caf/user/emiglior/SLHCSimPhase2/phase2/out81Xpre3/step2_2023tilted_v2.root')
)

# TAG
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Output
process.TFileService = cms.Service('TFileService',
    fileName = cms.string('file:cluster_validation.root')
)

# DEBUG
process.MessageLogger = cms.Service('MessageLogger',
	debugModules = cms.untracked.vstring('siPhase2Clusters'),
	destinations = cms.untracked.vstring('cout'),
	cout = cms.untracked.PSet(
		threshold = cms.untracked.string('ERROR')
	)
)

# Analyzer
process.analysis = cms.EDAnalyzer('Phase2TrackerClusterizerValidation',
    src = cms.InputTag("phase2ITPixelClusters"),
    links = cms.InputTag("simSiPixelDigis","Pixel")
)

# Processes to run
process.p = cms.Path(process.analysis)
