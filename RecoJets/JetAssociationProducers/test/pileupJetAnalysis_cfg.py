import FWCore.ParameterSet.Config as cms

process = cms.Process("Ana")

process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring(*[("rfio:/castor/cern.ch/user/t/tboccali/ttbar_fastsim_224/TTbar_cfi_GEN_FASTSIM_PU.root.%d" % (x + 1)) for x in range(20)])
)

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_V11::All'

process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

process.pileupJetTagger = cms.EDProducer("JetSignalVertexCompatibility.cc",
	jetTracksAssoc = cms.InputTag("ic5JetTracksAssociatorAtVertex"),
	primaryVertices = cms.InputTag("offlinePrimaryVertices"),
	cut = cms.double(3.0),
	temperature = cms.double(1.5)
)

process.pileupJetAnalyzer = cms.EDAnalyzer("PileupJetAnalyzer",
	jetTracksAssoc = cms.InputTag("ic5JetTracksAssociatorAtVertex"),
	jetTagLabel = cms.InputTag("pileupJetTagger"),
	signalFraction = cms.double(0.5),
	jetMinE = cms.double(10),
	jetMinEt = cms.double(10),
	jetMaxEta = cms.double(2.7),
	trackPtLimit = cms.double(100)
)

process.TFileService = cms.Service("TFileService",
	fileName = cms.string("analysis.root")
)

process.path = cms.Path(
	process.pileupJetTagger *
	process.pileupJetAnalyzer
)
