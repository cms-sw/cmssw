import FWCore.ParameterSet.Config as cms

hltMuonTauAnalyzer = cms.EDAnalyzer("HLTMuonTauAnalyzer",
    disableROOToutput = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True),
    RootFileName = cms.untracked.string("DQM_Higgs_214.root"),
    InputLabel = cms.untracked.InputTag("Muons"),
    TriggerCollection = cms.VPSet(
          L1CollectionLabel = cms.InputTag("hltMuonTauL1Filtered"),
          HLTCollectionLabels = cms.VInputTag(
            "hltMuonTauIsoL2IsoFiltered", "hltMuonTauIsoL2PreFiltered"
          , "hltMuonTauIsoL3IsoFiltered", "hltMuonTauIsoL3PreFiltered"
         ),
          L1ReferenceThreshold = cms.double(7.) ,
          HLTReferenceThreshold = cms.double(7.) ,
          NumberOfObjects = cms.uint32(1)
         ),
    NSigmas90 = cms.untracked.vdouble( 3., 3., 3., 3. ),
    CrossSection = cms.double(.97) ,
    Luminosity = cms.untracked.double(2.e30) ,
    PtMin = cms.untracked.double(0.0) ,
    PtMax = cms.untracked.double(100.0) ,
    Nbins = cms.untracked.uint32(40)
)
