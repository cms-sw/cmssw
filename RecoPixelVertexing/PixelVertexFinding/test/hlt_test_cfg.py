import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = 'GR09_R_34X_V5::All'
process.GlobalTag.globaltag = 'STARTUP3X_V8P::All'

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.source = cms.Source("PoolSource", 
	fileNames = cms.untracked.vstring(
        '/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8P_900GeV_Jan29ReReco-v1/0018/3E3776F8-550D-DF11-B137-00E0817917F3.root'
        ),
	debugVerbosity = cms.untracked.uint32(0),
	debugFlag = cms.untracked.bool(True)
)

process.test = cms.EDAnalyzer("PixelVertexTest",
	TrackCollection = cms.string("pixelTracks"),
	OutputTree = cms.untracked.string("pixel_vertexes_ttbar_HLT.root"),
	Verbosity = cms.untracked.uint32(0),
	simG4 = cms.InputTag("g4SimHits")
)

process.p = cms.Path(process.siPixelRecHits * process.pixelTracks * process.pixelVertices * process.test)


