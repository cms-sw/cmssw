import FWCore.ParameterSet.Config as cms

process = cms.Process("ZMuMuMCanalysis")
process.load("ElectroWeakAnalysis.ZReco.mcTruthForDimuons_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_1.root', 
        'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_2.root', 
        'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_3.root', 
        'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_4.root', 
        'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_6.root', 
        'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_7.root', 
        'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_8.root', 
        'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_9.root', 
        'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/zToMuMuWithoutFilter/degrutto/Zmumu/zToMuMuWithoutFilter/49ffac29c1b9792022f4be1caf28e358/dimuons_10.root'

    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('zMuMu_MCanalysis.root')
)

process.zToMuMu = cms.EDFilter("CandViewRefSelector",
    src = cms.InputTag("dimuons"),
    cut = cms.string('daughter(0).isGlobalMuon = 1 & daughter(1).isGlobalMuon = 1'),
)

process.goodZToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("dimuonsOneStandAloneMuon"),
    overlap = cms.InputTag("zToMuMu"),
)

process.zToMuMuOneTrack = cms.EDFilter(
    "CandViewRefSelector",
    src = cms.InputTag("dimuonsOneTrack"),
    cut = cms.string('daughter(0).isGlobalMuon = 1'),
)

process.goodZToMuMuOneTrack = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("zToMuMuOneTrack"),
    overlap = cms.InputTag("zToMuMu"),
)


process.zMuMu_MCanalyzer = cms.EDFilter("ZMuMu_MCanalyzer",
    muons = cms.InputTag("selectedLayer1Muons"),
    tracks = cms.InputTag("selectedLayer1TrackCands"),
    zMuMu = cms.InputTag("zToMuMu"),
    zMuStandAlone = cms.InputTag("goodZToMuMuOneStandAloneMuon"),
    zMuTrack = cms.InputTag("goodZToMuMuOneTrack"),
    zMuMuMatchMap = cms.InputTag("allDimuonsMCMatch"),
    zMuStandAloneMatchMap = cms.InputTag("allDimuonsMCMatch"),
    zMuTrackMatchMap = cms.InputTag("allDimuonsMCMatch"),
    genParticles = cms.InputTag("genParticles"),
    bothMuons = cms.bool(True),                              
    zMassMin = cms.untracked.double(20.0),
    zMassMax = cms.untracked.double(200.0),
    isomax = cms.untracked.double(3.0),
    etamin = cms.untracked.double(0.0),                                        
    etamax = cms.untracked.double(2.0),
    ptmin = cms.untracked.double(20.0),
)

process.eventInfo = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.mcTruthForDimuons *
                     process.zToMuMu *
                     process.goodZToMuMuOneStandAloneMuon *                     
                     process.zToMuMuOneTrack *
                     process.goodZToMuMuOneTrack *                     
                     process.zMuMu_MCanalyzer)
process.e = cms.EndPath(process.eventInfo)

