import FWCore.ParameterSet.Config as cms

l1PhaseIITree = cms.EDAnalyzer("L1PhaseIITreeProducer",

   jetToken = cms.untracked.InputTag("simCaloStage2Digis"),
   muonToken = cms.untracked.InputTag("simGmtStage2Digis"),
   sumToken = cms.untracked.InputTag("simCaloStage2Digis"),
   tauTokens = cms.untracked.VInputTag("simCaloStage2Digis"),

   egTokenBarrel = cms.InputTag("L1EGammaClusterEmuProducer","L1EGammaCollectionBXVEmulator"),
   tkEGTokenBarrel = cms.InputTag("L1TkElectronsCrystal","EG"),
   tkEGLooseTokenBarrel = cms.InputTag("L1TkElectronsLooseCrystal","EG"),
   tkEMTokenBarrel = cms.InputTag("L1TkPhotonsCrystal","EG"),

   egTokenHGC = cms.InputTag("l1EGammaEEProducer","L1EGammaCollectionBXVWithCuts"),
   tkEGTokenHGC = cms.InputTag("L1TkElectronsHGC","EG"),
   tkEGLooseTokenHGC = cms.InputTag("L1TkElectronsLooseHGC","EG"),
   tkEMTokenHGC = cms.InputTag("L1TkPhotonsHGC","EG"),

   tkTauToken = cms.InputTag("L1TkTauFromCalo",""), # ?
   TkGlbMuonToken = cms.InputTag("L1TkGlbMuons",""),
   TkMuonToken = cms.InputTag("L1TkMuons",""),                                            
   TkMuonStubsTokenBMTF = cms.InputTag("l1KBmtfStubMatchedMuons",""),
   TkMuonStubsTokenEMTF = cms.InputTag("l1TkMuonStubEndCap",""),

   tkTrackerJetToken = cms.InputTag("L1TrackerJets","L1TrackerJets"),                                            
   tkCaloJetToken = cms.InputTag("L1TkCaloJets","L1TkCaloJets"),
   tkMetToken = cms.InputTag("L1TrackerEtMiss","MET"),
   tkMhtTokens = cms.VInputTag( cms.InputTag("L1TrackerHTMiss5GeV","L1TrackerHTMiss"),cms.InputTag("L1TrackerHTMiss10GeV","L1TrackerHTMiss"),cms.InputTag("L1TrackerHTMiss","L1TrackerHTMiss"),cms.InputTag("L1TrackerHTMiss20GeV","L1TrackerHTMiss"),cms.InputTag("L1TrackerHTMiss30GeV","L1TrackerHTMiss")),

   ak4L1PF = cms.InputTag("ak4PFL1PuppiCorrected"),
   ak4L1PFForMET = cms.InputTag("ak4PFL1PuppiForMETCorrected"),
   l1PFCandidates = cms.InputTag("l1pfCandidates","Puppi"),

   caloJetToken = cms.InputTag("L1CaloJetProducer","L1CaloJetCollectionBXV"),
 
   muonKalman = cms.InputTag("simKBmtfDigis","BMTF"),
   muonOverlap = cms.InputTag("simOmtfDigis","OMTF"),
   muonEndcap = cms.InputTag("simEmtfDigis","EMTF"),

   l1PFMet = cms.InputTag("l1PFMetPuppiForMET"),

   zoPuppi = cms.InputTag("l1pfProducerBarrel","z0"),
   l1vertextdr = cms.InputTag("VertexProducer","l1vertextdr"),
   l1vertices = cms.InputTag("VertexProducer","l1vertices"),
   l1TkPrimaryVertex= cms.InputTag("L1TkPrimaryVertex",""),

   L1PFTauToken = cms.InputTag("l1pfTauProducer","L1PFTaus"),   

   maxL1Extra = cms.uint32(20)
)

#### Gen level tree

from L1Trigger.L1TNtuples.l1GeneratorTree_cfi  import l1GeneratorTree
genTree=l1GeneratorTree.clone()

runmenutree=cms.Path(l1PhaseIITree*genTree)


#### Additional collections that right now only the menu team is using for tuning - and that need to be cleaned!


from L1Trigger.L1TTrackMatch.L1TkHTMissProducer_cfi import L1TkCaloHTMiss

L1TrackerHTMiss5GeV = L1TkCaloHTMiss.clone()
L1TrackerHTMiss5GeV.L1TkJetInputTag = cms.InputTag("L1TrackerJets","L1TrackerJets")
L1TrackerHTMiss5GeV.jet_maxEta = cms.double(2.4)
L1TrackerHTMiss5GeV.jet_minPt = cms.double(5.0)
L1TrackerHTMiss5GeV.UseCaloJets = cms.bool(False)

L1TrackerHTMiss10GeV = L1TkCaloHTMiss.clone()
L1TrackerHTMiss10GeV.L1TkJetInputTag = cms.InputTag("L1TrackerJets","L1TrackerJets")
L1TrackerHTMiss10GeV.jet_maxEta = cms.double(2.4)
L1TrackerHTMiss10GeV.jet_minPt = cms.double(10.0)
L1TrackerHTMiss10GeV.UseCaloJets = cms.bool(False)

L1TrackerHTMiss20GeV = L1TkCaloHTMiss.clone()
L1TrackerHTMiss20GeV.L1TkJetInputTag = cms.InputTag("L1TrackerJets","L1TrackerJets")
L1TrackerHTMiss20GeV.jet_maxEta = cms.double(2.4)
L1TrackerHTMiss20GeV.jet_minPt = cms.double(20.0)
L1TrackerHTMiss20GeV.UseCaloJets = cms.bool(False)

L1TrackerHTMiss30GeV = L1TkCaloHTMiss.clone()
L1TrackerHTMiss30GeV.L1TkJetInputTag = cms.InputTag("L1TrackerJets","L1TrackerJets")
L1TrackerHTMiss30GeV.jet_maxEta = cms.double(2.4)
L1TrackerHTMiss30GeV.jet_minPt = cms.double(30.0)
L1TrackerHTMiss30GeV.UseCaloJets = cms.bool(False)


extraCollectionsMenuTree=cms.Path(L1TrackerHTMiss5GeV*L1TrackerHTMiss10GeV*L1TrackerHTMiss20GeV*L1TrackerHTMiss30GeV)


