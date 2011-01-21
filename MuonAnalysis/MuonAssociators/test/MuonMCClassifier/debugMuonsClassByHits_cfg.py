import FWCore.ParameterSet.Config as cms

process = cms.Process("PATMuon")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START38_V12::All'

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'root://pcmssd12.cern.ch//data/gpetrucc/7TeV/mu11/store/mc/Fall10/QCD_Pt_600to800_TuneZ2_7TeV_pythia6/GEN-SIM-RECO/E7TeV_ProbDist_2010Data_BX156_START38_V12-v1/0096/22B09051-20EA-DF11-84C6-00151796C1E8.root',
        #'root://pcmssd12.cern.ch//data/gpetrucc/7TeV/mu11/store/mc/Fall10/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/GEN-SIM-RECO/E7TeV_ProbDist_2010Data_BX156_START38_V12-v1/0070/F044B493-E3E5-DF11-ACBD-001A92811728.root'
    ),
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.selMuons = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("pt > 15 && isGlobalMuon"),
    filter = cms.bool(True),
)

process.load("MuonAnalysis.MuonAssociators.muonClassificationByHits_cfi")

process.classByHits = process.classByHitsGlb.clone(muons = "selMuons", muonPreselection = "")

process.go = cms.Path(
    process.selMuons    +
    process.mix +
    process.trackingParticlesNoSimHits +
    process.classByHits
)

process.MessageLogger.categories += [ 'MuonMCClassifier' ]
process.MessageLogger.cerr.MuonMCClassifier = cms.untracked.PSet(
    optionalPSet = cms.untracked.bool(True),
    limit = cms.untracked.int32(10000000)
)

