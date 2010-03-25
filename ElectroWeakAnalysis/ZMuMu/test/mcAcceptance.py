import FWCore.ParameterSet.Config as cms

process = cms.Process("MCAcceptance")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(13000)
)

process.include("FWCore/MessageLogger/data/MessageLogger.cfi")

process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    'rfio:/castor/cern.ch/user/f/fabozzi/mc7tev/F8EE38AF-1EBE-DE11-8D19-00304891F14E.root'

#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/38980FEC-C182-DE11-A3B5-003048D4767C.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/3AF703B9-AE82-DE11-9656-0015172C0925.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/46854F8E-BC82-DE11-80AA-003048D47673.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/8025F9B0-AC82-DE11-8C28-0015172560C6.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/88DDF58E-BC82-DE11-ADD8-003048D47679.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/9A115324-BB82-DE11-9C66-001517252130.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/Zmumu7TeV/GEN-SIM_RECO/0014/FC279CAC-AD82-DE11-BAAA-001517357D36.root")
)
)
process.evtInfo = cms.OutputModule("AsciiOutputModule")




process.zToMuMuMC = cms.EDFilter("CandViewRefSelector",
    src = cms.InputTag("genParticles"),
    cut = cms.string('pdgId = 23 & status = 3 & abs(daughter(0).pdgId) = 13')
)

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START3X_V21::All')
process.load("Configuration.StandardSequences.MagneticField_cff")


process.goodMuons = cms.EDFilter("MuonSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('isGlobalMuon = 1'),
  filter = cms.bool(False)                                
)

#


process.dimuons = cms.EDFilter("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string('mass > 0'),
    decay = cms.string('goodMuons@+ goodMuons@-')
)




process.load("PhysicsTools.HepMCCandAlgos.goodMuonMCMatch_cfi")
process.goodMuonMCMatch.src = 'goodMuons'


process.dimuonsMCMatch = cms.EDFilter("MCTruthCompositeMatcherNew",
    src = cms.InputTag("dimuons"),
    #
    # comment PAT match because works only for layer-0 muons  
    #
    #  VInputTag matchMaps = { muonMatch }
    matchPDGId = cms.vint32(),
    matchMaps = cms.VInputTag(cms.InputTag("goodMuonMCMatch"))
)




process.mcAcceptance = cms.EDAnalyzer("MCAcceptanceAnalyzer",
    zToMuMu = cms.InputTag("dimuons"),
    zToMuMuMC = cms.InputTag("zToMuMuMC"),
    zToMuMuMatched = cms.InputTag("dimuonsMCMatch"),
    massMin = cms.double(60.0),
    massMax = cms.double(120.0),
    etaDau0Min = cms.double(2.1),
    etaDau0Max = cms.double(2.4),
    etaDau1Min = cms.double(2.1),
    etaDau1Max = cms.double(2.4),                                  
    ptMin = cms.double(20.0),
    # parameter for denominator
    massMinZMC = cms.double(60.0),
    massMaxZMC = cms.double(120.0)                                       
)

process.mcPath = cms.Path(
    process.zToMuMuMC +
    process.goodMuons +
    process.goodMuonMCMatch +
    process.dimuons +
    process.dimuonsMCMatch+ 
    process.mcAcceptance
    )

from Configuration.EventContent.EventContent_cff import *

process.EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_dimuons_*_*',
        'keep *_goodMuons_*_*',
        'keep *_genParticles_*_*',
        'keep *_goodMuonMCMatch_*_*', 
        'keep *_dimuonsMCMatch_*_*', 
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
   fileName = cms.untracked.string('dimuons_forAcceptance_1000.root')
)




process.end = cms.EndPath(process.dimuonsOutputModule)


