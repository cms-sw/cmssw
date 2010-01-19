import FWCore.ParameterSet.Config as cms

process = cms.Process("MCAcceptance")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/0ABB0814-C082-DE11-9AB7-003048D4767C.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/38980FEC-C182-DE11-A3B5-003048D4767C.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/3AF703B9-AE82-DE11-9656-0015172C0925.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/46854F8E-BC82-DE11-80AA-003048D47673.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/8025F9B0-AC82-DE11-8C28-0015172560C6.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/88DDF58E-BC82-DE11-ADD8-003048D47679.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/9A115324-BB82-DE11-9C66-001517252130.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/FC279CAC-AD82-DE11-BAAA-001517357D36.root")
)

process.evtInfo = cms.OutputModule("AsciiOutputModule")




process.zToMuMuMC = cms.EDFilter("CandViewRefSelector",
    src = cms.InputTag("genParticles"),
    cut = cms.string('pdgId = 23 & status = 3 & abs(daughter(0).pdgId) = 13')
)

## process.dimuons = cms.EDFilter("CandViewShallowCloneCombiner",
##     checkCharge = cms.bool(False),
##     cut = cms.string('mass > 0'),
##     #  string decay = "goodMuons@+ goodMuons@-"
## #    decay = cms.string('selectedLayer1Muons@+ selectedLayer1Muons@-')
##     decay = cms.string('muons@+ muons@-')
## )


process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V3::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("ElectroWeakAnalysis.Skimming.dimuons_cfi")

process.load("ElectroWeakAnalysis.Skimming.dimuonsOneTrack_cfi")

process.load("ElectroWeakAnalysis.Skimming.mcTruthForDimuons_cff")
#from ElectroWeakAnalysis.Skimming.dimuons_cfi import *
process.load("ElectroWeakAnalysis.Skimming.patCandidatesForDimuonsSequences_cff")

#from ElectroWeakAnalysis.Skimming.patCandidatesForDimuonsSequences_cff import *


process.mcAcceptance = cms.EDAnalyzer("MCAcceptanceAnalyzer",
    zToMuMu = cms.InputTag("dimuons"),
    zToMuMuMC = cms.InputTag("zToMuMuMC"),
    zToMuMuMatched = cms.InputTag("dimuonsMCMatch"),
    massMin = cms.double(60.0),
    massMax = cms.double(120.0),
    etaMin = cms.double(0.0),
    etaMax = cms.double(2.1),
    ptMin = cms.double(20.0)
    
)

process.mcPath = cms.Path(
    process.zToMuMuMC+
    process.goodMuonRecoForDimuon *
    process.dimuons *
    process.dimuonsOneTrack *
    process.mcTruthForDimuons *

 #   process.dimuons +
 #   process.dimuonMatches
    process.mcAcceptance
    )

from Configuration.EventContent.EventContent_cff import *

process.EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_dimuonsGlobal_*_*', 
        'keep *_dimuonsOneStandAloneMuon_*_*', 
        'keep *_muonMatch_*_*', 
        'keep *_trackMuMatch_*_*', 
        'keep *_allDimuonsMCMatch_*_*',
#        'keep patTriggerObjects_patTrigger_*_*',
#        'keep patTriggerFilters_patTrigger_*_*',
#        'keep patTriggerPaths_patTrigger_*_*',
#        'keep patTriggerEvent_patTriggerEvent_*_*',
#        'keep patTriggerObjectsedmAssociation_patTriggerEvent_*_*'
        )
)

AODSIMDimuonEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMDimuonEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMDimuonEventContent.outputCommands.extend(process.EventContent.outputCommands)

process.dimuonsOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMDimuonEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('acceptance'),
        dataTier = cms.untracked.string('USER')
   ),
   fileName = cms.untracked.string('dimuons.root')
)




process.end = cms.EndPath(process.dimuonsOutputModule)


