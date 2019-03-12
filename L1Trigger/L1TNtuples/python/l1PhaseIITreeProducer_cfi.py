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

   tkTrackerJetToken = cms.InputTag("TwoLayerJets", "L1TwoLayerJets"),                                            
   tkCaloJetToken = cms.InputTag("L1TkCaloJets","L1TkCaloJets"),
   tkMetToken = cms.InputTag("L1TrackerEtMiss","trkMET"),
   tkMhtTokens = cms.VInputTag( cms.InputTag("L1TrackerHTMiss","L1TrackerHTMiss") ),

   ak4L1PF = cms.InputTag("ak4PFL1PuppiCorrected"),
#   ak4L1PFForMET = cms.InputTag("ak4PFL1PuppiForMETCorrected"),
   l1PFCandidates = cms.InputTag("l1pfCandidates","Puppi"),

   caloJetToken = cms.InputTag("L1CaloJetProducer","L1CaloJetCollectionBXV"),
 
   muonKalman = cms.InputTag("simKBmtfDigis","BMTF"),
   muonOverlap = cms.InputTag("simOmtfDigis","OMTF"),
   muonEndcap = cms.InputTag("simEmtfDigis","EMTF"),

   l1PFMet = cms.InputTag("l1PFMetPuppi"),

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




