import FWCore.ParameterSet.Config as cms

process = cms.Process("MCAcceptance")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'file:/scratch2/users/fabozzi/spring10/powheg_zmm_cteq66/1CD6D0A6-1E64-DF11-BB60-001D09FD0D10.root',
'file:/scratch2/users/fabozzi/spring10/powheg_zmm_cteq66/46BC4EF2-B462-DF11-8FE0-0015178C4D14.root',
'file:/scratch2/users/fabozzi/spring10/powheg_zmm_cteq66/48C1BEAB-1E64-DF11-8874-A4BADB3CF509.root',
'file:/scratch2/users/fabozzi/spring10/powheg_zmm_cteq66/9E650E9F-1963-DF11-B7CE-0024E8768D1A.root',
'file:/scratch2/users/fabozzi/spring10/powheg_zmm_cteq66/C28E2383-1D64-DF11-894B-A4BADB3CF8F5.root',
'file:/scratch2/users/fabozzi/spring10/powheg_zmm_cteq66/D21A4219-1D64-DF11-A4EC-001D09FD0D10.root',
'file:/scratch2/users/fabozzi/spring10/powheg_zmm_cteq66/F0D65D50-F162-DF11-93BB-A4BADB3CF208.root',
'file:/scratch2/users/fabozzi/spring10/powheg_zmm_cteq66/F4A6969A-9562-DF11-8192-00E08142962E.root',
'file:/scratch2/users/fabozzi/spring10/powheg_zmm_cteq66/F8CAC083-6462-DF11-9D7E-001C23C0E208.root',
'file:/scratch2/users/fabozzi/spring10/powheg_zmm_cteq66/FCD6E07C-9562-DF11-9C9A-001EC94BA3D1.root',



)
)
process.evtInfo = cms.OutputModule("AsciiOutputModule")




process.zToMuMuMC = cms.EDFilter("CandViewRefSelector",
    src = cms.InputTag("genParticles"),
    cut = cms.string('pdgId = 23 & status = 3 & abs(daughter(0).pdgId) = 13')
)

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_36Y_V10::All')
process.load("Configuration.StandardSequences.MagneticField_cff")


process.goodMuons = cms.EDFilter("MuonSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('isGlobalMuon = 1 && pt>15.'),
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
process.goodMuonMCMatch.maxDPtRel = cms.double(0.1)
process.goodMuonMCMatch.resolveByMatchQuality = cms.bool(True)
process.goodMuonMCMatch.maxDeltaR = cms.double(0.1)
process.goodMuonMCMatch.checkCharge = cms.bool(True)
process.goodMuonMCMatch.resolveAmbiguities = cms.bool(True)




process.dimuonsMCMatch = cms.EDFilter("MCTruthCompositeMatcherNew",
    src = cms.InputTag("dimuons"),
    #
    # comment PAT match because works only for layer-0 muons  
    #
    #  VInputTag matchMaps = { muonMatch }
    matchPDGId = cms.vint32(23,13),
    matchMaps = cms.VInputTag(cms.InputTag("goodMuonMCMatch"))
)




process.mcAcceptanceAA = cms.EDAnalyzer("MCAcceptanceAnalyzer",
    zToMuMu = cms.InputTag("dimuons"),
    zToMuMuMC = cms.InputTag("zToMuMuMC"),
    zToMuMuMatched = cms.InputTag("dimuonsMCMatch"),
    massMin = cms.double(60.0),
    massMax = cms.double(120.0),
    etaDau0Min = cms.double(0.),
    etaDau0Max = cms.double(2.1),
    etaDau1Min = cms.double(0.),
    etaDau1Max = cms.double(2.1),                                  
    ptMin = cms.double(20.0),
    massMinZMC = cms.double(60.0),
    massMaxZMC = cms.double(120.0)                                       
)

#process.mcAcceptanceBA = cms.EDAnalyzer("MCAcceptanceAnalyzer",
#    zToMuMu = cms.InputTag("dimuons"),
#    zToMuMuMC = cms.InputTag("zToMuMuMC"),
#    zToMuMuMatched = cms.InputTag("dimuonsMCMatch"),
#    massMin = cms.double(60.0),
#    massMax = cms.double(120.0),
#    etaDau0Min = cms.double(2.1),
#    etaDau0Max = cms.double(2.4),
#    etaDau1Min = cms.double(0.),
#    etaDau1Max = cms.double(2.1),                                  
#    ptMin = cms.double(20.0),
#    massMinZMC = cms.double(60.0),
#    massMaxZMC = cms.double(120.0)                                       
#)


process.mcAcceptanceBar = cms.EDAnalyzer("MCAcceptanceAnalyzer",
    zToMuMu = cms.InputTag("dimuons"),
    zToMuMuMC = cms.InputTag("zToMuMuMC"),
    zToMuMuMatched = cms.InputTag("dimuonsMCMatch"),
    massMin = cms.double(60.0),
    massMax = cms.double(120.0),
    etaDau0Min = cms.double(0.),
    etaDau0Max = cms.double(0.9),
    etaDau1Min = cms.double(0.),
    etaDau1Max = cms.double(2.4),                                  
    ptMin = cms.double(20.0),
    massMinZMC = cms.double(60.0),
    massMaxZMC = cms.double(120.0)                                       
)

process.mcAcceptanceBarEnd = cms.EDAnalyzer("MCAcceptanceAnalyzer",
    zToMuMu = cms.InputTag("dimuons"),
    zToMuMuMC = cms.InputTag("zToMuMuMC"),
    zToMuMuMatched = cms.InputTag("dimuonsMCMatch"),
    massMin = cms.double(60.0),
    massMax = cms.double(120.0),
    etaDau0Min = cms.double(0.9),
    etaDau0Max = cms.double(1.2),
    etaDau1Min = cms.double(0.),
    etaDau1Max = cms.double(2.4),                                  
    ptMin = cms.double(20.0),
    massMinZMC = cms.double(60.0),
    massMaxZMC = cms.double(120.0)                                       
)



process.mcAcceptanceEnd = cms.EDAnalyzer("MCAcceptanceAnalyzer",
    zToMuMu = cms.InputTag("dimuons"),
    zToMuMuMC = cms.InputTag("zToMuMuMC"),
    zToMuMuMatched = cms.InputTag("dimuonsMCMatch"),
    massMin = cms.double(60.0),
    massMax = cms.double(120.0),
    etaDau0Min = cms.double(1.2),
    etaDau0Max = cms.double(2.1),
    etaDau1Min = cms.double(0.),
    etaDau1Max = cms.double(2.4),                                  
    ptMin = cms.double(20.0),
    massMinZMC = cms.double(60.0),
    massMaxZMC = cms.double(120.0)                                       
)






process.mcPath = cms.Path(
    process.zToMuMuMC +
    process.goodMuons +
    process.goodMuonMCMatch +
    process.dimuons +
    process.dimuonsMCMatch+ 
    process.mcAcceptanceAA 
  #  process.mcAcceptanceBar +
  #  process.mcAcceptanceBarEnd +
  #  process.mcAcceptanceEnd 
    
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


