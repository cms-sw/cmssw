import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.common_cff import *
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
            skipObjectsNotPassingQualityBits = cms.bool(True),
            qualityBits = cms.string(
                            "filter('*CaloIdLTrackIdLIsoVL*TrackIso*Filter') + " \
                            "2*filter('hltEle*WPTight*TrackIsoFilter*') + " \
                            "4*filter('hltEle*WPLoose*TrackIsoFilter') + " \
                            "8*filter('*OverlapFilter*IsoEle*PFTau*') + " \
                            "16*filter('hltEle*Ele*CaloIdLTrackIdLIsoVL*Filter') + " \
                            "32*filter('hltMu*TrkIsoVVL*Ele*CaloIdLTrackIdLIsoVL*Filter*')  + " \
                            "64*filter('hltOverlapFilterIsoEle*PFTau*') + " \
                            "128*filter('hltEle*Ele*Ele*CaloIdLTrackIdLDphiLeg*Filter') + " \
                            "256*max(filter('hltL3fL1Mu*DoubleEG*Filtered*'),filter('hltMu*DiEle*CaloIdLTrackIdLElectronleg*Filter')) + " \
                            "512*max(filter('hltL3fL1DoubleMu*EG*Filter*'),filter('hltDiMu*Ele*CaloIdLTrackIdLElectronleg*Filter')) + " \
                            "1024*min(filter('hltEle32L1DoubleEGWPTightGsfTrackIsoFilter'),filter('hltEGL1SingleEGOrFilter')) + " \
                            "2048*filter('hltEle*CaloIdVTGsfTrkIdTGsfDphiFilter') + " \
                            "4096*path('HLT_Ele*PFJet*') + " \
                            "8192*max(filter('hltEG175HEFilter'),filter('hltEG200HEFilter'))"),
            qualityBitsDoc = cms.string("1 = CaloIdL_TrackIdL_IsoVL, 2 = 1e (WPTight), 4 = 1e (WPLoose), 8 = OverlapFilter PFTau, 16 = 2e, 32 = 1e-1mu, 64 = 1e-1tau, 128 = 3e, 256 = 2e-1mu, 512 = 1e-2mu, 1024 = 1e (32_L1DoubleEG_AND_L1SingleEGOr), 2048 = 1e (CaloIdVT_GsfTrkIdT), 4096 = 1e (PFJet), 8192 = 1e (Photon175_OR_Photon200)"),
            ),
        cms.PSet(
            name = cms.string("Photon (PixelMatch-vetoed e/gamma)"), 
            id = cms.int32(22),
            sel = cms.string("type(92) && pt > 20 && coll('hltEgammaCandidates') && !filter('*PixelMatchFilter')"), 
            l1seed = cms.string("type(-98)"),  l1deltaR = cms.double(0.3),
            #l2seed = cms.string("type(92) && coll('')"),  l2deltaR = cms.double(0.5),
            skipObjectsNotPassingQualityBits = cms.bool(True),
            qualityBits = cms.string("0"), qualityBitsDoc = cms.string(""),
        ),
        cms.PSet(
            name = cms.string("Muon"),
            id = cms.int32(13),
            sel = cms.string("type(83) && pt > 5 && (coll('hltIterL3MuonCandidates') || (pt > 45 && coll('hltHighPtTkMuonCands')) || (pt > 95 && coll('hltOldL3MuonCandidates')))"),
            l1seed = cms.string("type(-81)"), l1deltaR = cms.double(0.5),
            l2seed = cms.string("type(83) && coll('hltL2MuonCandidates')"),  l2deltaR = cms.double(0.3),
            skipObjectsNotPassingQualityBits = cms.bool(True),
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
                            "512*max(filter('hltL3fL1Mu*DoubleEG*Filtered*'),filter('hltMu*DiEle*CaloIdLTrackIdLElectronleg*Filter')) + " \
                            "1024*max(filter('hltL3fL1sMu*L3Filtered50*'),filter('hltL3fL1sMu*TkFiltered50*')) + " \
                            "2048*max(filter('hltL3fL1sMu*L3Filtered100*'),filter('hltL3fL1sMu*TkFiltered100*'))"),
            qualityBitsDoc = cms.string("1 = TrkIsoVVL, 2 = Iso, 4 = OverlapFilter PFTau, 8 = 1mu, 16 = 2mu, 32 = 1mu-1e, 64 = 1mu-1tau, 128 = 3mu, 256 = 2mu-1e, 512 = 1mu-2e, 1024 = 1mu (Mu50), 2048 = 1mu (Mu100)"),
            ),
        cms.PSet(
            name = cms.string("Tau"),
            id = cms.int32(15),
            sel = cms.string("type(84) && pt > 5 && coll('*Tau*') && ( filter('*LooseChargedIso*') || filter('*MediumChargedIso*') || filter('*TightChargedIso*') || filter('*TightOOSCPhotons*') || filter('hltL2TauIsoFilter') || filter('*OverlapFilterIsoMu*') || filter('*OverlapFilterIsoEle*') || filter('*L1HLTMatched*') || filter('*Dz02*') )"), #All trigger objects from a Tau collection + passing at least one filter
            l1seed = cms.string("type(-100)"), l1deltaR = cms.double(0.3),
            l2seed = cms.string("type(84) && coll('hltL2TauJetsL1IsoTauSeeded')"),  l2deltaR = cms.double(0.3),
            skipObjectsNotPassingQualityBits = cms.bool(True),
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
            sel = cms.string("( type(0) || type(85) || type(86) || type(-99) )"), 
            l1seed = cms.string("type(-99)"), l1deltaR = cms.double(0.3),
            l2seed = cms.string("type(85) || type(86) || type(-99)"),  l2deltaR = cms.double(0.3),
            skipObjectsNotPassingQualityBits = cms.bool(True),
            qualityBits = cms.string(
                "1         * filter('*CrossCleaned*LooseChargedIsoPFTau*') + " \
                "2         * filter('hltBTagCaloCSVp087Triple') + " \
                "4         * filter('hltDoubleCentralJet90') + " \
                "8         * filter('hltDoublePFCentralJetLooseID90') + " \
                "16        * filter('hltL1sTripleJetVBFIorHTTIorDoubleJetCIorSingleJet') + " \
                "32        * filter('hltQuadCentralJet30') + " \
                "64        * filter('hltQuadPFCentralJetLooseID30') + " \
                "128       * max(filter('hltL1sQuadJetC50IorQuadJetC60IorHTT280IorHTT300IorHTT320IorTripleJet846848VBFIorTripleJet887256VBFIorTripleJet927664VBF'), filter('hltL1sQuadJetCIorTripleJetVBFIorHTT')) + " \
                "256       * filter('hltQuadCentralJet45') + " \
                "512       * filter('hltQuadPFCentralJetLooseID45') + " \
                "1024      * max(filter('hltL1sQuadJetC60IorHTT380IorHTT280QuadJetIorHTT300QuadJet'), filter('hltL1sQuadJetC50to60IorHTT280to500IorHTT250to340QuadJet')) + " \
                "2048      * max(filter('hltBTagCaloCSVp05Double'), filter('hltBTagCaloDeepCSVp17Double')) + " \
                "4096      * filter('hltPFCentralJetLooseIDQuad30') + " \
                "8192      * filter('hlt1PFCentralJetLooseID75') + " \
                "16384     * filter('hlt2PFCentralJetLooseID60') + " \
                "32768     * filter('hlt3PFCentralJetLooseID45') + " \
                "65536     * filter('hlt4PFCentralJetLooseID40') + " \
                "131072    * max(filter('hltBTagPFCSVp070Triple'), max(filter('hltBTagPFDeepCSVp24Triple'), filter('hltBTagPFDeepCSV4p5Triple')) )"
                ), 
            qualityBitsDoc = cms.string(
                "Jet bits: bit 0 for VBF cross-cleaned from loose iso PFTau, bit 1 for hltBTagCaloCSVp087Triple, bit 2 for hltDoubleCentralJet90, bit 3 for hltDoublePFCentralJetLooseID90," \
                " bit 4 for hltL1sTripleJetVBFIorHTTIorDoubleJetCIorSingleJet, bit 5 for hltQuadCentralJet30, bit 6 for hltQuadPFCentralJetLooseID30," \
                " bit 7 for hltL1sQuadJetC50IorQuadJetC60IorHTT280IorHTT300IorHTT320IorTripleJet846848VBFIorTripleJet887256VBFIorTripleJet927664VBF or hltL1sQuadJetCIorTripleJetVBFIorHTT," \
                " bit 8 for hltQuadCentralJet45, bit 9 for hltQuadPFCentralJetLooseID45," \
                " bit 10  for hltL1sQuadJetC60IorHTT380IorHTT280QuadJetIorHTT300QuadJet or hltL1sQuadJetC50to60IorHTT280to500IorHTT250to340QuadJet" \
                " bit 11 for hltBTagCaloCSVp05Double or hltBTagCaloDeepCSVp17Double, bit 12 for hltPFCentralJetLooseIDQuad30, bit 13 for hlt1PFCentralJetLooseID75," \
                " bit 14 for hlt2PFCentralJetLooseID60, bit 15 for hlt3PFCentralJetLooseID45, bit 16 for hlt4PFCentralJetLooseID40," \
                " bit 17 for hltBTagPFCSVp070Triple or hltBTagPFDeepCSVp24Triple or hltBTagPFDeepCSV4p5Triple "),
        ),
        cms.PSet(
            name = cms.string("FatJet"),
            id = cms.int32(6),
            sel = cms.string("type(85) && pt > 120 && coll('hltAK8PFJetsCorrected')"), 
            l1seed = cms.string("type(-99)"), l1deltaR = cms.double(0.3),
            l2seed = cms.string("type(85)  && coll('hltAK8CaloJetsCorrectedIDPassed')"),  l2deltaR = cms.double(0.3),
            skipObjectsNotPassingQualityBits = cms.bool(True),
            qualityBits = cms.string("0"), qualityBitsDoc = cms.string(""),
        ),
        cms.PSet(
            name = cms.string("MET"),
            id = cms.int32(2),
            sel = cms.string("type(87) && pt > 30 && coll('hltPFMETProducer')"), 
            l1seed = cms.string("type(-87) && coll('L1ETM')"), l1deltaR = cms.double(9999),
            l1seed_2 = cms.string("type(-87) && coll('L1ETMHF')"), l1deltaR_2 = cms.double(9999),
            l2seed = cms.string("type( 87) && coll('hltMetClean')"),  l2deltaR = cms.double(9999),
            skipObjectsNotPassingQualityBits = cms.bool(True),
            qualityBits = cms.string("0"), qualityBitsDoc = cms.string(""),
        ),
        cms.PSet(
            name = cms.string("HT"),
            id = cms.int32(3),
            sel = cms.string("type(89) || type(-89)"), 
            l1seed = cms.string("type(-89) && coll('L1HTT')"), l1deltaR = cms.double(9999),
            l1seed_2 = cms.string("type(-89) && coll('L1HTTHF')"), l1deltaR_2 = cms.double(9999),
            l2seed = cms.string("type(89) && coll('hltHtMhtJet30')"),  l2deltaR = cms.double(9999),
            skipObjectsNotPassingQualityBits = cms.bool(True),
            qualityBits = cms.string(
                "1        * filter('hltL1sTripleJetVBFIorHTTIorDoubleJetCIorSingleJet') + " \
                "2        * max(filter('hltL1sQuadJetC50IorQuadJetC60IorHTT280IorHTT300IorHTT320IorTripleJet846848VBFIorTripleJet887256VBFIorTripleJet927664VBF'), filter('hltL1sQuadJetCIorTripleJetVBFIorHTT')) + " \
                "4        * max(filter('hltL1sQuadJetC60IorHTT380IorHTT280QuadJetIorHTT300QuadJet'), filter('hltL1sQuadJetC50to60IorHTT280to500IorHTT250to340QuadJet')) + " \
                "8        * max(filter('hltCaloQuadJet30HT300'), filter('hltCaloQuadJet30HT320')) + " \
                "16       * max(filter('hltPFCentralJetsLooseIDQuad30HT300'), filter('hltPFCentralJetsLooseIDQuad30HT330'))"
                ), 
            qualityBitsDoc = cms.string(
                "HT bits: bit 0 for hltL1sTripleJetVBFIorHTTIorDoubleJetCIorSingleJet, bit 1 for hltL1sQuadJetC50IorQuadJetC60IorHTT280IorHTT300IorHTT320IorTripleJet846848VBFIorTripleJet887256VBFIorTripleJet927664VBF or hltL1sQuadJetCIorTripleJetVBFIorHTT, " \
                "bit 2 for hltL1sQuadJetC60IorHTT380IorHTT280QuadJetIorHTT300QuadJet or hltL1sQuadJetC50to60IorHTT280to500IorHTT250to340QuadJet, " \
                "bit 3 for hltCaloQuadJet30HT300 or hltCaloQuadJet30HT320, bit 4 for hltPFCentralJetsLooseIDQuad30HT300 or hltPFCentralJetsLooseIDQuad30HT330"
            ),
        ),
        cms.PSet(
            name = cms.string("MHT"),
            id = cms.int32(4),
            sel = cms.string("type(90)"), 
            l1seed = cms.string("type(-90) && coll('L1HTM')"), l1deltaR = cms.double(9999),
            l1seed_2 = cms.string("type(-90) && coll('L1HTMHF')"), l1deltaR_2 = cms.double(9999),
            l2seed = cms.string("type(90) && coll('hltHtMhtJet30')"),  l2deltaR = cms.double(9999),
            skipObjectsNotPassingQualityBits = cms.bool(True),
            qualityBits = cms.string(
                "1       * max(filter('hltCaloQuadJet30HT300'), filter('hltCaloQuadJet30HT320')) + " \
                "2       * max(filter('hltPFCentralJetsLooseIDQuad30HT300'), filter('hltPFCentralJetsLooseIDQuad30HT330'))"
                ), 
                qualityBitsDoc = cms.string
                (
                "MHT bits: bit 0 for hltCaloQuadJet30HT300 or hltCaloQuadJet30HT320, bit 1 for hltPFCentralJetsLooseIDQuad30HT300 or hltPFCentralJetsLooseIDQuad30HT330"
            ),
        ),

    ),
)

# ERA-dependent configuration
# Tune filter and collection names to 2016 HLT menus
# FIXME: check non-lepton objects and cross check leptons
selections2016 = copy.deepcopy(triggerObjectTable.selections)
for sel in selections2016:
    if sel.name=='Muon':
        sel.sel = cms.string("type(83) && pt > 5 && (coll('hlt*L3MuonCandidates') || coll('hlt*TkMuonCands') || coll('hlt*TrkMuonCands'))")
        sel.qualityBits = cms.string("filter('*RelTrkIso*Filtered0p4') + 2*filter('hltL3cr*IsoFiltered0p09') + 4*filter('*OverlapFilter*IsoMu*PFTau*') + 8*filter('hltL3f*IsoFiltered0p09') + 1024*max(filter('hltL3fL1sMu*L3Filtered50*'),filter('hltL3fL1sMu*TkFiltered50*'))")
        sel.qualityBitsDoc = cms.string("1 = TrkIsoVVL, 2 = Iso, 4 = OverlapFilter PFTau, 8 = IsoTkMu, 1024 = 1mu (Mu50)")
    elif sel.name=='Tau':
        sel.sel = cms.string("type(84) && pt > 5 && coll('*Tau*') && ( filter('*LooseIso*') || filter('*MediumIso*') || filter('*MediumComb*Iso*') || filter('hltL2TauIsoFilter') || filter('*OverlapFilter*IsoMu*') || filter('*OverlapFilter*IsoEle*') || filter('*L1HLTMatched*') || filter('*Dz02*') )")
        sel.qualityBits = cms.string("(filter('*LooseIso*')-filter('*VLooseIso*'))+2*filter('*Medium*Iso*')+4*filter('*VLooseIso*')+8*0+16*filter('hltL2TauIsoFilter')+32*filter('*OverlapFilter*IsoMu*')+64*filter('*OverlapFilter*IsoEle*')+128*filter('*L1HLTMatched*')+256*filter('*Dz02*')")
        sel.qualityBitsDoc = cms.string("1 = LooseIso, 2 = Medium(Comb)Iso, 4 = VLooseIso, 8 = None, 16 = L2p5 pixel iso, 32 = OverlapFilter IsoMu, 64 = OverlapFilter IsoEle, 128 = L1-HLT matched, 256 = Dz")

run2_HLTconditions_2016.toModify(
  triggerObjectTable,
  selections = selections2016
)

from PhysicsTools.PatUtils.L1PrefiringWeightProducer_cff import prefiringweight
#Next lines are for UL2016 maps
(run2_muon_2016 & tracker_apv_vfp30_2016).toModify( prefiringweight, DataEraECAL = cms.string("UL2016preVFP"),  DataEraMuon = cms.string("2016preVFP"))
(run2_muon_2016 & ~tracker_apv_vfp30_2016).toModify( prefiringweight, DataEraECAL = cms.string("UL2016postVFP"),  DataEraMuon = cms.string("2016postVFP"))
#Next line is for UL2017 maps 
run2_jme_2017.toModify( prefiringweight, DataEraECAL = cms.string("UL2017BtoF"), DataEraMuon = cms.string("20172018"))
#Next line is for UL2018 maps 
run2_muon_2018.toModify( prefiringweight, DataEraECAL = cms.string("None"), DataEraMuon = cms.string("20172018"))

#For pre-UL 2017 reprocessing, one should use the original maps and no muon jet protection  
for modifier in run2_nanoAOD_94XMiniAODv1, run2_nanoAOD_94XMiniAODv2:
    modifier.toModify( prefiringweight, DataEraECAL = cms.string("2017BtoF"), DataEraMuon = cms.string("20172018"))
    modifier.toModify( prefiringweight, JetMaxMuonFraction = cms.double(-1.) )
#For pre-UL 2016 reprocessing, same thing
run2_nanoAOD_94X2016.toModify( prefiringweight, DataEraECAL = cms.string("2016BtoH"), DataEraMuon = cms.string("2016") )
run2_nanoAOD_94X2016.toModify( prefiringweight, JetMaxMuonFraction = cms.double(-1.) )

l1PreFiringEventWeightTable = cms.EDProducer("GlobalVariablesTableProducer",
    name = cms.string("L1PreFiringWeight"),
    variables = cms.PSet(
        Nom = ExtVar(cms.InputTag("prefiringweight:nonPrefiringProb"), "float", doc = "L1 pre-firing event correction weight (1-probability)", precision=8),
        Up = ExtVar(cms.InputTag("prefiringweight:nonPrefiringProbUp"), "float", doc = "L1 pre-firing event correction weight (1-probability), up var.", precision=8),
        Dn = ExtVar(cms.InputTag("prefiringweight:nonPrefiringProbDown"), "float", doc = "L1 pre-firing event correction weight (1-probability), down var.", precision=8),
        Muon_Nom = ExtVar(cms.InputTag("prefiringweight:nonPrefiringProbMuon"), "float", doc = "Muon L1 pre-firing event correction weight (1-probability)", precision=8),
        Muon_SystUp = ExtVar(cms.InputTag("prefiringweight:nonPrefiringProbMuonSystUp"), "float", doc = "Muon L1 pre-firing event correction weight (1-probability), up var. syst.", precision=8),
        Muon_SystDn = ExtVar(cms.InputTag("prefiringweight:nonPrefiringProbMuonSystDown"), "float", doc = "Muon L1 pre-firing event correction weight (1-probability), down var. syst.", precision=8),
        Muon_StatUp = ExtVar(cms.InputTag("prefiringweight:nonPrefiringProbMuonStatUp"), "float", doc = "Muon L1 pre-firing event correction weight (1-probability), up var. stat.", precision=8),
        Muon_StatDn = ExtVar(cms.InputTag("prefiringweight:nonPrefiringProbMuonStatDown"), "float", doc = "Muon L1 pre-firing event correction weight (1-probability), down var. stat.", precision=8),
        ECAL_Nom = ExtVar(cms.InputTag("prefiringweight:nonPrefiringProbECAL"), "float", doc = "ECAL L1 pre-firing event correction weight (1-probability)", precision=8),
        ECAL_Up = ExtVar(cms.InputTag("prefiringweight:nonPrefiringProbECALUp"), "float", doc = "ECAL L1 pre-firing event correction weight (1-probability), up var.", precision=8),
        ECAL_Dn = ExtVar(cms.InputTag("prefiringweight:nonPrefiringProbECALDown"), "float", doc = "ECAL L1 pre-firing event correction weight (1-probability), down var.", precision=8),
    )
)

triggerObjectTables = cms.Sequence( unpackedPatTrigger + triggerObjectTable )

_triggerObjectTables_withL1PreFiring = triggerObjectTables.copy()
_triggerObjectTables_withL1PreFiring.replace(triggerObjectTable, prefiringweight + l1PreFiringEventWeightTable + triggerObjectTable)
(run2_HLTconditions_2016 | run2_HLTconditions_2017 | run2_HLTconditions_2018).toReplaceWith(triggerObjectTables, _triggerObjectTables_withL1PreFiring)
