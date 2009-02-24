import FWCore.ParameterSet.Config as cms

process = cms.Process("MCAcceptance")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_1.root', 
        'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_2.root', 
        'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_3.root', 
        'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_4.root', 
        'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_6.root', 
        'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_7.root', 
        'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_8.root', 
        'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_9.root', 
        'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_10.root')
)

process.evtInfo = cms.OutputModule("AsciiOutputModule")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('mcAcceptanceStudy.root')
)

process.zToMuMuMC = cms.EDFilter("CandViewRefSelector",
    src = cms.InputTag("genParticles"),
    cut = cms.string('pdgId = 23 & status = 3 & abs(daughter(0).pdgId) = 13')
)

process.dimuonMatches = cms.EDFilter("DimuonMCMatcher",
    src = cms.InputTag("dimuons")
)

process.mcAcceptance = cms.EDAnalyzer("MCAcceptanceAnalyzer",
    zToMuMu = cms.InputTag("dimuons"),
    zToMuMuMC = cms.InputTag("zToMuMuMC"),
    zToMuMuMatched = cms.InputTag("dimuonMatches"),
    massMax = cms.double(120.0),
    etaMin = cms.double(0.0),
    etaMax = cms.double(0.8),
    massMin = cms.double(60.0),
    ptMin = cms.double(20.0)
    
)

process.mcPath = cms.Path(
    process.zToMuMuMC+
    process.dimuonMatches+
    process.mcAcceptance
    )
process.end = cms.EndPath(process.evtInfo)


