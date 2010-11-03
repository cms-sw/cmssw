import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("TCTauAOD")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#	'rfio:/castor/cern.ch/user/s/slehti/testData/Fall10_QCD_Pt_80to120_Tune1_7TeV_pythia8_GEN-SIM-RECO_START38_V12-v1_0007_360E0EDF-49CB-DF11-AEEC-E0CB4E55364D.root'
	'rfio:/castor/cern.ch/user/s/slehti/testData/Fall10_QCD_Pt_120to170_TuneZ2_7TeV_pythia6_AODSIM_START38_V12-v1_0003_D094F01B-E6CB-DF11-9C2F-003048D46028.root'
#	"file:testAOD.root"
    )
)

process.load("FWCore/MessageService/MessageLogger_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'STARTUP31X_V1::All'
#process.GlobalTag.globaltag = cms.string('GR09_R_34X_V2::All')
process.GlobalTag.globaltag = cms.string('START38_V13::All')

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

#process.load("JetMETCorrections/TauJet/TCTauProducer_cff")
process.load("RecoTauTag.Configuration.RecoTauTag_cff")

process.runTCTauProducer = cms.Path(
    process.TCTau
)

process.load("Configuration.EventContent.EventContent_cff")
process.TESTOUT = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
        dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('AODSIM'),
        dataTier = cms.untracked.string('AODSIM')
    ),
#    outputCommands = cms.untracked.vstring(
#        "keep *"
#    ),
    fileName = cms.untracked.string('file:testout.root')
)
process.outpath = cms.EndPath(process.TESTOUT)
