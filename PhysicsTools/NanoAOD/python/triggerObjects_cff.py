import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
import copy

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
    l1EG = cms.InputTag("caloStage2Digis","EGamma"),
    l1Sum = cms.InputTag("caloStage2Digis","EtSum"),
    l1Jet = cms.InputTag("caloStage2Digis","Jet"),
    l1Muon = cms.InputTag("gmtStage2Digis","Muon"),
    l1Tau = cms.InputTag("caloStage2Digis","Tau"),
    selections = cms.VPSet(
        cms.PSet(
            name = cms.string("Electron (PixelMatched e/gamma)"), # this selects also photons for the moment!
            id = cms.int32(11),
            sel = cms.string("type(92) && pt > 7 && coll('hltEgammaCandidates') && filter('*PixelMatchFilter')"), 
            l1seed = cms.string("type(-98)"),  l1deltaR = cms.double(0.3),
            #l2seed = cms.string("type(92) && coll('')"),  l2deltaR = cms.double(0.5),
            qualityBits = cms.string(
                              "filter('*CaloIdLTrackIdLIsoVL*TrackIso*Filter') + " \
                              "2*filter('hltEle*WPTight*TrackIsoFilter*') + " \
                              "4*filter('hltEle*WPLoose*TrackIsoFilter') + " \
                              "8*filter('*OverlapFilterIsoEle*PFTau*') + " \
                              "16*filter('hltEle*Ele*CaloIdLTrackIdLIsoVL*Filter') + " \
                              "32*filter('hltMu*TrkIsoVVL*Ele*CaloIdLTrackIdLIsoVL*Filter*')  + " \
                              "64*filter('*OverlapFilterIsoEle*PFTau*') + " \
                              "128*filter('hltEle*Ele*Ele*CaloIdLTrackIdLDphiLeg*Filter') + " \
                              "256*max(filter('hltL3fL1Mu*DoubleEG*Filtered*'),filter('hltMu*DiEle*CaloIdLTrackIdLElectronleg*Filter')) + " \
                              "512*max(filter('hltL3fL1DoubleMu*EG*Filter*'),filter('hltDiMu*Ele*CaloIdLTrackIdLElectronleg*Filter'))"),
            qualityBitsDoc = cms.string("1 = CaloIdL_TrackIdL_IsoVL, 2 = 1e (WPTight), 4 = 1e (WPLoose), 8 = OverlapFilter PFTau, 16 = 2e, 32 = 1e-1mu, 64 = 1e-1tau, 128 = 3e, 256 = 2e-1mu, 512 = 1e-2mu"),
            ),
        cms.PSet(
            name = cms.string("Photon (PixelMatch-vetoed e/gamma)"), 
            id = cms.int32(22),
            sel = cms.string("type(92) && pt > 20 && coll('hltEgammaCandidates') && !filter('*PixelMatchFilter')"), 
            l1seed = cms.string("type(-98)"),  l1deltaR = cms.double(0.3),
            #l2seed = cms.string("type(92) && coll('')"),  l2deltaR = cms.double(0.5),
            qualityBits = cms.string("0"), qualityBitsDoc = cms.string(""),
        ),
        cms.PSet(
            name = cms.string("Muon"),
            id = cms.int32(13),
            sel = cms.string("type(83) && pt > 5 && coll('hltIterL3MuonCandidates')"), 
            l1seed = cms.string("type(-81)"), l1deltaR = cms.double(0.5),
            l2seed = cms.string("type(83) && coll('hltL2MuonCandidates')"),  l2deltaR = cms.double(0.3),
            qualityBits = cms.string(
                            "filter('*RelTrkIsoVVLFiltered0p4') + " \
                            "2*filter('hltL3crIso*Filtered0p07') + " \
                            "4*filter('*OverlapFilterIsoMu*PFTau*') + " \
                            "8*max(filter('hltL3crIsoL1*SingleMu*Filtered0p07'),filter('hltL3crIsoL1sMu*Filtered0p07')) + " \
                            "16*filter('hltDiMuon*Filtered*') + " \
                            "32*filter('hltMu*TrkIsoVVL*Ele*CaloIdLTrackIdLIsoVL*Filter*') + " \
                            "64*filter('hltOverlapFilterIsoMu*PFTau*') + " \
                            "128*filter('hltL3fL1TripleMu*') + " \
                            "256*max(filter('hltL3fL1DoubleMu*EG*Filtered*'),filter('hltDiMu*Ele*CaloIdLTrackIdLElectronleg*Filter')) + " \
                            "512*max(filter('hltL3fL1Mu*DoubleEG*Filtered*'),filter('hltMu*DiEle*CaloIdLTrackIdLElectronleg*Filter'))"),    
            qualityBitsDoc = cms.string("1 = TrkIsoVVL, 2 = Iso, 4 = OverlapFilter PFTau, 8 = 1mu, 16 = 2mu, 32 = 1mu-1e, 64 = 1mu-1tau, 128 = 3mu, 256 = 2mu-1e, 512 =1mu-2e"),
            ),
        cms.PSet(
            name = cms.string("Tau"),
            id = cms.int32(15),
            sel = cms.string("type(84) && pt > 5 && coll('*Tau*') && ( filter('*LooseChargedIso*') || filter('*MediumChargedIso*') || filter('*TightChargedIso*') || filter('*TightOOSCPhotons*') || filter('hltL2TauIsoFilter') || filter('*OverlapFilterIsoMu*') || filter('*OverlapFilterIsoEle*') || filter('*L1HLTMatched*') || filter('*Dz02*') )"), #All trigger objects from a Tau collection + passing at least one filter
            l1seed = cms.string("type(-100)"), l1deltaR = cms.double(0.3),
            l2seed = cms.string("type(84) && coll('hltL2TauJetsL1IsoTauSeeded')"),  l2deltaR = cms.double(0.3),
            qualityBits = cms.string(
                               "filter('*LooseChargedIso*') + " \
                               "2*filter('*MediumChargedIso*') + " \
                               "4*filter('*TightChargedIso*') + " \
                               "8*filter('*TightOOSCPhotons*') + " \
                               "16*filter('*Hps*') + " \
                               "32*filter('hltSelectedPFTau*MediumChargedIsolationL1HLTMatched*') + " \
                               "64*filter('hltDoublePFTau*TrackPt1*ChargedIsolation*Dz02Reg') + " \
                               "128*filter('hltOverlapFilterIsoEle*PFTau*') + " \
                               "256*filter('hltOverlapFilterIsoMu*PFTau*') + " \
                               "512*filter('hltDoublePFTau*TrackPt1*ChargedIsolation*')"),
            qualityBitsDoc = cms.string("1 = LooseChargedIso, 2 = MediumChargedIso, 4 = TightChargedIso, 8 = TightID OOSC photons, 16 = HPS, 32 = single-tau + tau+MET, 64 = di-tau, 128 = e-tau, 256 = mu-tau, 512 = VBF+di-tau"),            
        ),   
        cms.PSet(
            name = cms.string("Jet"),
            id = cms.int32(1),
            sel = cms.string("type(85) && pt > 30 && (coll('hltAK4PFJetsCorrected') || coll('hltMatchedVBF*PFJets*PFTau*OverlapRemoval'))"), 
            l1seed = cms.string("type(-99)"), l1deltaR = cms.double(0.3),
            l2seed = cms.string("type(85)  && coll('hltAK4CaloJetsCorrectedIDPassed')"),  l2deltaR = cms.double(0.3),
            qualityBits = cms.string("filter('*CrossCleaned*LooseChargedIsoPFTau*')"), qualityBitsDoc = cms.string("1 = VBF cross-cleaned from loose iso PFTau"),
        ),
        cms.PSet(
            name = cms.string("FatJet"),
            id = cms.int32(6),
            sel = cms.string("type(85) && pt > 120 && coll('hltAK8PFJetsCorrected')"), 
            l1seed = cms.string("type(-99)"), l1deltaR = cms.double(0.3),
            l2seed = cms.string("type(85)  && coll('hltAK8CaloJetsCorrectedIDPassed')"),  l2deltaR = cms.double(0.3),
            qualityBits = cms.string("0"), qualityBitsDoc = cms.string(""),
        ),
        cms.PSet(
            name = cms.string("MET"),
            id = cms.int32(2),
            sel = cms.string("type(87) && pt > 30 && coll('hltPFMETProducer')"), 
            l1seed = cms.string("type(-87) && coll('L1ETM')"), l1deltaR = cms.double(9999),
            l1seed_2 = cms.string("type(-87) && coll('L1ETMHF')"), l1deltaR_2 = cms.double(9999),
            l2seed = cms.string("type( 87) && coll('hltMetClean')"),  l2deltaR = cms.double(9999),
            qualityBits = cms.string("0"), qualityBitsDoc = cms.string(""),
        ),
        cms.PSet(
            name = cms.string("HT"),
            id = cms.int32(3),
            sel = cms.string("type(89) && pt > 100 && coll('hltPFHTJet30')"), 
            l1seed = cms.string("type(-89) && coll('L1HTT')"), l1deltaR = cms.double(9999),
            l1seed_2 = cms.string("type(-89) && coll('L1HTTHF')"), l1deltaR_2 = cms.double(9999),
            #l2seed = cms.string("type(89) && coll('hltHtMhtJet30')"),  l2deltaR = cms.double(9999),
            qualityBits = cms.string("0"), qualityBitsDoc = cms.string(""),
        ),
        cms.PSet(
            name = cms.string("MHT"),
            id = cms.int32(4),
            sel = cms.string("type(90) && pt > 30 && coll('hltPFMHTTightID')"), 
            l1seed = cms.string("type(-90) && coll('L1HTM')"), l1deltaR = cms.double(9999),
            l1seed_2 = cms.string("type(-90) && coll('L1HTMHF')"), l1deltaR_2 = cms.double(9999),
            #l2seed = cms.string("type(90) && coll('hltHtMhtJet30')"),  l2deltaR = cms.double(9999),
            qualityBits = cms.string("0"), qualityBitsDoc = cms.string(""),
        ),

    ),
)

# ERA-dependent configuration
# Tune filter and collection names to 2016 HLT menus
# FIXME: check non-lepton objects and cross check leptons
selections80X = copy.deepcopy(triggerObjectTable.selections)
for sel in selections80X:
    if sel.name=='Muon':
        sel.sel = cms.string("type(83) && pt > 5 && (coll('hlt*L3MuonCandidates') || coll('hlt*TkMuonCands') || coll('hlt*TrkMuonCands'))")
        sel.qualityBits = cms.string("filter('*RelTrkIso*Filtered0p4') + 2*filter('hltL3cr*IsoFiltered0p09') + 4*filter('*OverlapFilter*IsoMu*PFTau*') + 8*filter('hltL3f*IsoFiltered0p09')")
        sel.qualityBitsDoc = cms.string("1 = TrkIsoVVL, 2 = Iso, 4 = OverlapFilter PFTau, 8 = IsoTkMu")
    elif sel.name=='Tau':
        sel.sel = cms.string("type(84) && pt > 5 && coll('*Tau*') && ( filter('*LooseIso*') || filter('*MediumIso*') || filter('*MediumComb*Iso*') || filter('hltL2TauIsoFilter') || filter('*OverlapFilter*IsoMu*') || filter('*OverlapFilter*IsoEle*') || filter('*L1HLTMatched*') || filter('*Dz02*') )")
        sel.qualityBits = cms.string("(filter('*LooseIso*')-filter('*VLooseIso*'))+2*filter('*Medium*Iso*')+4*filter('*VLooseIso*')+8*0+16*filter('hltL2TauIsoFilter')+32*filter('*OverlapFilter*IsoMu*')+64*filter('*OverlapFilter*IsoEle*')+128*filter('*L1HLTMatched*')+256*filter('*Dz02*')")
        sel.qualityBitsDoc = cms.string("1 = LooseIso, 2 = Medium(Comb)Iso, 4 = VLooseIso, 8 = None, 16 = L2p5 pixel iso, 32 = OverlapFilter IsoMu, 64 = OverlapFilter IsoEle, 128 = L1-HLT matched, 256 = Dz")
    elif sel.name=='Electron (PixelMatched e/gamma)':
        #sel.sel = cms.string("type(92) && pt > 7 && coll('hltEgammaCandidates') && filter('*PixelMatchFilter')")
        sel.qualityBits = cms.string("filter('*CaloIdLTrackIdLIsoVL*TrackIso*Filter') + 2*filter('hltEle*WPTight*TrackIsoFilter*') + 4*filter('hltEle*WPLoose*TrackIsoFilter') + 8*filter('*OverlapFilter*IsoEle*PFTau*')")
        #sel.qualityBitsDoc = cms.string("1 = CaloIdL_TrackIdL_IsoVL, 2 = WPLoose, 4 = WPTight, 8 = OverlapFilter PFTau")

run2_miniAOD_80XLegacy.toModify(
  triggerObjectTable,
  selections = selections80X
)

triggerObjectTables = cms.Sequence( unpackedPatTrigger + triggerObjectTable )
