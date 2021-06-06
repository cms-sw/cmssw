import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTX")

### Load all ESSources, ESProducers and PSets
# process.load("HLTrigger.Configuration.Phase2.hltPhase2Setup_cff")

### GlobalTag
# process.load("Configuration.StandardSequences.CondDBESSource_cff")
# process.GlobalTag.globaltag = "112X_mcRun4_realistic_T15_v2"

process.l1tEle7 = cms.EDFilter(
    "L1TTkEleFilter",
    MinPt=cms.double(7.0),
    MinEta=cms.double(-2.4),
    MaxEta=cms.double(2.4),
    inputTag1=cms.InputTag("L1TkElectronsEllipticMatchCrystal", "EG"),
    inputTag2=cms.InputTag("L1TkElectronsEllipticMatchHGC", "EG"),
    Scalings=cms.PSet(
        barrel=cms.vdouble(0.805095, 1.18336, 0.0),
        endcap=cms.vdouble(0.453144, 1.26205, 0.0),
    ),
    EtaBinsForIsolation=cms.vdouble(0.0, 9999.9),
    TrkIsolation=cms.vdouble(99999.9),  # No isolation
    ApplyQual1=cms.bool(True),
    ApplyQual2=cms.bool(True),
    Quality1=cms.int32(2),  # 0x2
    Quality2=cms.int32(5),
    Qual1IsMask=cms.bool(True),
    Qual2IsMask=cms.bool(False),
)

process.l1tIsoEle7 = cms.EDFilter(
    "L1TTkEleFilter",
    MinPt=cms.double(7.0),
    MinEta=cms.double(-2.4),
    MaxEta=cms.double(2.4),
    inputTag1=cms.InputTag("L1TkElectronsEllipticMatchCrystal", "EG"),
    inputTag2=cms.InputTag("L1TkElectronsEllipticMatchHGC", "EG"),
    Scalings=cms.PSet(
        barrel=cms.vdouble(0.434262, 1.20586, 0.0),
        endcap=cms.vdouble(0.266186, 1.25976, 0.0),
    ),
    EtaBinsForIsolation=cms.vdouble(0.0, 1.479, 9999.9),
    TrkIsolation=cms.vdouble(0.12, 0.2),
    ApplyQual1=cms.bool(False),
    ApplyQual2=cms.bool(True),
    Quality1=cms.int32(-1),
    Quality2=cms.int32(5),
    Qual1IsMask=cms.bool(False),
    Qual2IsMask=cms.bool(False),
)

process.l1tIsoPho7 = cms.EDFilter(
    "L1TTkEmFilter",
    MinPt=cms.double(7.0),
    MinEta=cms.double(-2.4),
    MaxEta=cms.double(2.4),
    inputTag1=cms.InputTag("L1TkPhotonsCrystal", "EG"),
    inputTag2=cms.InputTag("L1TkPhotonsHGC", "EG"),
    Scalings=cms.PSet(
        barrel=cms.vdouble(2.54255, 1.08749, 0.0),
        endcap=cms.vdouble(2.11186, 1.15524, 0.0),
    ),
    EtaBinsForIsolation=cms.vdouble(0.0, 1.479, 9999.9),
    TrkIsolation=cms.vdouble(0.28, 0.35),
    ApplyQual1=cms.bool(False),
    ApplyQual2=cms.bool(True),
    Quality1=cms.int32(2),  # 0x2 "second bit"
    Quality2=cms.int32(5),
    Qual1IsMask=cms.bool(True),
    Qual2IsMask=cms.bool(False),
)

process.l1tMuon7 = cms.EDFilter(
    "L1TTkMuonFilter",
    MinPt=cms.double(7.0),
    MinEta=cms.double(-2.4),
    MaxEta=cms.double(2.4),
    inputTag=cms.InputTag("L1TkMuons"),
    Scalings=cms.PSet(
        barrel=cms.vdouble(0.802461, 1.04193, 0.0),
        overlap=cms.vdouble(0.921315, 1.03611, 0.0),
        endcap=cms.vdouble(0.828802, 1.03447, 0.0),
    ),
)

process.l1tDoubleMuon7 = cms.EDFilter(
    "L1TTkMuonFilter",
    MinN=cms.int32(2),
    MinPt=cms.double(7.0),
    MinEta=cms.double(-2.4),
    MaxEta=cms.double(2.4),
    inputTag=cms.InputTag("L1TkMuons"),
    Scalings=cms.PSet(
        barrel=cms.vdouble(0.802461, 1.04193, 0.0),
        overlap=cms.vdouble(0.921315, 1.03611, 0.0),
        endcap=cms.vdouble(0.828802, 1.03447, 0.0),
    ),
)

process.l1tDoubleMuon7DZ0p33 = cms.EDFilter(
    "HLT2L1TkMuonL1TkMuonDZ",
    originTag1=cms.VInputTag(
        "L1TkMuons",
    ),
    originTag2=cms.VInputTag(
        "L1TkMuons",
    ),
    inputTag1=cms.InputTag("l1tDoubleMuon7"),
    inputTag2=cms.InputTag("l1tDoubleMuon7"),
    triggerType1=cms.int32(-114),  # L1TkMuon
    triggerType2=cms.int32(-114),  # L1TkMuon
    MinDR=cms.double(-1),
    MaxDZ=cms.double(0.33),
    MinPixHitsForDZ=cms.int32(0),  # Immaterial
    checkSC=cms.bool(False),  # Immaterial
    MinN=cms.int32(1),
)

process.l1tPFJet64 = cms.EDFilter(
    "L1TPFJetFilter",
    inputTag=cms.InputTag("ak4PFL1PuppiCorrected"),
    Scalings=cms.PSet(
        barrel=cms.vdouble(11.1254, 1.40627, 0),
        overlap=cms.vdouble(24.8375, 1.4152, 0),
        endcap=cms.vdouble(42.4039, 1.33052, 0),
    ),
    MinPt=cms.double(64.0),
    MinEta=cms.double(-2.4),
    MaxEta=cms.double(2.4),
)

process.L1PFHtMht = cms.EDProducer(
    "HLTHtMhtProducer",
    jetsLabel=cms.InputTag("ak4PFL1PuppiCorrected"),
    minPtJetHt=cms.double(30),
    maxEtaJetHt=cms.double(2.4),
)

# ### Notice that there is no MHT seed in the Phase-II Level-1 Menu...
# # Possible choices for TypeOfSum are: MET, MHT, ETT, HT
# # but notice that if you are using a MET seed you
# # should probably use the precomputed one.

# # We don't have scaling for MHT...
process.l1tPFMht40 = cms.EDFilter(
    "L1TEnergySumFilter",
    inputTag=cms.InputTag("L1PFHtMht"),
    Scalings=cms.PSet(
        theScalings=cms.vdouble(0, 1, 0),
    ),
    TypeOfSum=cms.string("MHT"),
    MinPt=cms.double(40.0),
)

process.l1tPFHt90 = cms.EDFilter(
    "L1TEnergySumFilter",
    inputTag=cms.InputTag("L1PFHtMht"),
    Scalings=cms.PSet(
        # theScalings = cms.vdouble(-7.12716,1.03067,0), # PFPhase1HTOfflineEtCut
        theScalings=cms.vdouble(50.0182, 1.0961, 0),  # PFPhase1HT090OfflineEtCut
    ),
    TypeOfSum=cms.string("MHT"),
    MinPt=cms.double(90.0),
)

process.l1tPFMet90 = cms.EDFilter(
    "L1TPFEnergySumFilter",
    inputTag=cms.InputTag("l1PFMetPuppi"),
    Scalings=cms.PSet(
        # theScalings = cms.vdouble(-7.24159,1.20973,0), # PuppiMETOfflineEtCut
        theScalings=cms.vdouble(54.2859, 1.39739, 0),  # PuppiMET090OfflineEtCut
        # theScalings = cms.vdouble(0,0,0),
    ),
    TypeOfSum=cms.string("MET"),
    MinPt=cms.double(90.0),
)

process.HLT_SingleEle7 = cms.Path(process.l1tEle7)
process.HLT_SingleIsoEle7 = cms.Path(process.l1tIsoEle7)
process.HLT_SingleIsoPhoton7 = cms.Path(process.l1tIsoPho7)
process.HLT_SingleMu7 = cms.Path(process.l1tMuon7)
process.HLT_DoubleMu7_DZ0p33 = cms.Path(
    process.l1tDoubleMuon7 + process.l1tDoubleMuon7DZ0p33
)
process.HLT_SingleJet64 = cms.Path(process.l1tPFJet64)
process.HLT_MHT40 = cms.Path(process.L1PFHtMht + process.l1tPFMht40)
process.HLT_HT90 = cms.Path(process.L1PFHtMht + process.l1tPFHt90)
process.HLT_MET90 = cms.Path(process.l1tPFMet90)

process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(
        "/store/mc/Phase2HLTTDRSummer20ReRECOMiniAOD/DYToLL_M-50_TuneCP5_14TeV-pythia8/FEVT/PU200_pilot_111X_mcRun4_realistic_T15_v1-v1/270000/FF7BF0E2-1380-2D48-BB19-F79E6907CD5D.root",
        # "/store/mc/Phase2HLTTDRSummer20ReRECOMiniAOD/SingleElectron_PT2to200/FEVT/PU200_111X_mcRun4_realistic_T15_v1_ext2-v1/270000/0064D31F-F48B-3144-8CB9-17F820065E01.root",
    ),
)

process.maxEvents.input = cms.untracked.int32(-1)
process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(True))
