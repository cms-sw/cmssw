import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START3X_V16D::All'

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10000))

process.source = cms.Source("PoolSource", 
	fileNames = cms.untracked.vstring(
          '/store/mc/Summer09/MinBias/GEN-SIM-RECO/V16D_900GeV-v1/0001/FCD70794-F216-DF11-931A-0015170AC494.root',
          '/store/mc/Summer09/MinBias/GEN-SIM-RECO/V16D_900GeV-v1/0001/FAEF5289-F116-DF11-9862-003048D460F8.root',
          '/store/mc/Summer09/MinBias/GEN-SIM-RECO/V16D_900GeV-v1/0001/F6F40994-F216-DF11-88DB-0015170AE63C.root',
          '/store/mc/Summer09/MinBias/GEN-SIM-RECO/V16D_900GeV-v1/0001/F63CE52F-F216-DF11-B4AB-003048D479F0.root',
          '/store/mc/Summer09/MinBias/GEN-SIM-RECO/V16D_900GeV-v1/0001/F048EADE-F116-DF11-83D5-00E08178C119.root',
          '/store/mc/Summer09/MinBias/GEN-SIM-RECO/V16D_900GeV-v1/0001/ECADEDD2-2517-DF11-B839-00E08178C14F.root'
        )
)

process.Timing = cms.Service("Timing",
        useJobReport = cms.untracked.bool(True)
)

process.test = cms.EDAnalyzer("PixelVertexTest",
	TrackCollection = cms.string("pixelTracks"),
	OutputTree = cms.untracked.string("MC_MinBias.root"),
	Verbosity = cms.untracked.uint32(0),
	simG4 = cms.InputTag("g4SimHits")
)

process.options = cms.untracked.PSet(
        wantSummary = cms.untracked.bool(True)
)
process.p = cms.Path(process.siPixelRecHits * process.pixelTracks * process.pixelVertices * process.test)
