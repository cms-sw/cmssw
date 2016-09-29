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
process.load('Configuration.Geometry.GeometryExtended2023D3Reco_cff')
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
                            fileNames = cms.untracked.vstring('file:/tmp/emiglior/step3.root'),
                            secondaryFileNames=cms.untracked.vstring('file:/tmp/emiglior/step2.root')

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
	debugModules = cms.untracked.vstring('analysis_Clusters'),
	destinations = cms.untracked.vstring('cout'),
	cout = cms.untracked.PSet(
		threshold = cms.untracked.string('ERROR')
	)
)

# Analyzer
process.analysis_Cluster = cms.EDAnalyzer('Phase2ITPixelClustersValidation',
    src = cms.InputTag("phase2ITPixelClusters"),
    links = cms.InputTag("simSiPixelDigis","Pixel")
)

# Processes to run
process.p = cms.Path(process.analysis_Cluster)
