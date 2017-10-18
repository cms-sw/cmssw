import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy

unpackedPatTrigger = cms.EDProducer("PATTriggerObjectStandAloneUnpacker",
    patTriggerObjectsStandAlone = cms.InputTag('slimmedPatTrigger'),
    triggerResults              = cms.InputTag('TriggerResults::HLT'),
    unpackFilterLabels = cms.bool(True)
)
# ERA-dependent configuration
run2_miniAOD_80XLegacy.toModify(
  unpackedPatTrigger,
  patTriggerObjectsStandAlone = "selectedPatTrigger",
  unpackFilterLabels = False 
)

triggerObjectTable = cms.EDProducer("TriggerObjectTableProducer",
    name= cms.string("TrigObj"),
    src = cms.InputTag("unpackedPatTrigger"),
    selections = cms.VPSet(
        cms.PSet(
            name = cms.string("Electron (PixelMatched e/gamma)"), # this selects also photons for the moment!
            id = cms.int32(11),
            sel = cms.string("type(92) && pt > 7 && coll('hltEgammaCandidates') && filter('*PixelMatchFilter')"), 
            l1seed = cms.string("type(-98) && coll('hltGtStage2Digis:EGamma')"),  l1deltaR = cms.double(0.3),
            #l2seed = cms.string("type(92) && coll('')"),  l2deltaR = cms.double(0.5),
            qualityBits = cms.string("filter('*CaloIdLTrackIdLIsoVL*TrackIso*Filter') + 2*filter('hltEle*WPTight*TrackIsoFilter') + 4*filter('hltEle*WPLoose*TrackIsoFilter')"), 
            qualityBitsDoc = cms.string("1 = CaloIdL_TrackIdL_IsoVL, 2 = WPLoose, 4 = WPTight"),
        ),
        cms.PSet(
            name = cms.string("Photon (PixelMatch-vetoed e/gamma)"), 
            id = cms.int32(22),
            sel = cms.string("type(92) && pt > 20 && coll('hltEgammaCandidates') && !filter('*PixelMatchFilter')"), 
            l1seed = cms.string("type(-98) && coll('hltGtStage2Digis:EGamma')"),  l1deltaR = cms.double(0.3),
            #l2seed = cms.string("type(92) && coll('')"),  l2deltaR = cms.double(0.5),
            qualityBits = cms.string("0"), qualityBitsDoc = cms.string(""),
        ),
        cms.PSet(
            name = cms.string("Muon"),
            id = cms.int32(13),
            sel = cms.string("type(83) && pt > 5 && coll('hltIterL3MuonCandidates')"), 
            l1seed = cms.string("type(-81) && coll('hltGtStage2Digis:Muon')"), l1deltaR = cms.double(0.5),
            l2seed = cms.string("type(83) && coll('hltL2MuonCandidates')"),  l2deltaR = cms.double(0.3),
            qualityBits = cms.string("filter('*RelTrkIsoVVLFiltered0p4') + 2*filter('hltL3crIso*Filtered')"), qualityBitsDoc = cms.string("1 = TrkIsoVVL, 2 = Iso"),
        ),
        cms.PSet(
            name = cms.string("Tau"),
            id = cms.int32(14),
            sel = cms.string("type(84) && pt > 5 && coll('hltPFTaus')"), 
            l1seed = cms.string("type(-100) && coll('hltGtStage2Digis:Tau')"), l1deltaR = cms.double(0.3),
            l2seed = cms.string("type(84) && coll('hltL2TauJetsL1IsoTauSeeded')"),  l2deltaR = cms.double(0.3),
            qualityBits = cms.string("0"), qualityBitsDoc = cms.string(""),
        ),
        cms.PSet(
            name = cms.string("Jet"),
            id = cms.int32(1),
            sel = cms.string("type(85) && pt > 30 && coll('hltAK4PFJetsCorrected')"), 
            l1seed = cms.string("type(-99) && coll('hltGtStage2Digis:Jet')"), l1deltaR = cms.double(0.3),
            l2seed = cms.string("type(85)  && coll('hltAK4CaloJetsCorrectedIDPassed')"),  l2deltaR = cms.double(0.3),
            qualityBits = cms.string("0"), qualityBitsDoc = cms.string(""),
        ),
        cms.PSet(
            name = cms.string("MET"),
            id = cms.int32(2),
            sel = cms.string("type(87) && pt > 30 && coll('hltPFMETProducer')"), 
            l1seed = cms.string("type(-87) && coll('hltGtStage2Digis:EtSum')"), l1deltaR = cms.double(9999),
            l2seed = cms.string("type( 87) && coll('hltMetClean')"),  l2deltaR = cms.double(9999),
            qualityBits = cms.string("0"), qualityBitsDoc = cms.string(""),
        ),
        cms.PSet(
            name = cms.string("HT"),
            id = cms.int32(3),
            sel = cms.string("type(89) && pt > 100 && coll('hltPFHTJet30')"), 
            l1seed = cms.string("type(-89) && coll('hltGtStage2Digis:EtSum')"), l1deltaR = cms.double(9999),
            #l2seed = cms.string("type(89) && coll('hltHtMhtJet30')"),  l2deltaR = cms.double(9999),
            qualityBits = cms.string("0"), qualityBitsDoc = cms.string(""),
        ),
        cms.PSet(
            name = cms.string("MHT"),
            id = cms.int32(4),
            sel = cms.string("type(90) && pt > 30 && coll('hltPFMHTTightID')"), 
            l1seed = cms.string("type(-90) && coll('hltGtStage2Digis:EtSum')"), l1deltaR = cms.double(9999),
            #l2seed = cms.string("type(90) && coll('hltHtMhtJet30')"),  l2deltaR = cms.double(9999),
            qualityBits = cms.string("0"), qualityBitsDoc = cms.string(""),
        ),

    ),
)

triggerObjectTables = cms.Sequence( unpackedPatTrigger + triggerObjectTable )
