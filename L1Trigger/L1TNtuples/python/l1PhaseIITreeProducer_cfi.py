import FWCore.ParameterSet.Config as cms

l1PhaseIITree = cms.EDAnalyzer("L1PhaseIITreeProducer",

   jetToken = cms.untracked.InputTag("simCaloStage2Digis"),
   muonToken = cms.untracked.InputTag("simGmtStage2Digis"),
   sumToken = cms.untracked.InputTag("simCaloStage2Digis"),
   tauTokens = cms.untracked.VInputTag("simCaloStage2Digis"),

   egTokens = cms.VInputTag(cms.InputTag("l1EGammaCrystalsProducer","L1EGammaCollectionBXVWithCuts"),cms.InputTag("l1EGammaEEProducer","L1EGammaCollectionBXVWithCuts")),
   tkEGTokens = cms.VInputTag( cms.InputTag("L1TkElectronsCrystal","EG"),cms.InputTag("L1TkElectronsHGC","EG") ),
   tkEGLooseTokens = cms.VInputTag( cms.InputTag("L1TkElectronsLooseCrystal","EG"),cms.InputTag("L1TkElectronsLooseHGC","EG") ),
   tkEMTokens = cms.VInputTag( cms.InputTag("L1TkPhotonsCrystal","EG"),cms.InputTag("L1TkPhotonsHGC","EG") ),

   tkTauToken = cms.InputTag("L1TkTauFromCalo",""), # ?
   TkGlbMuonToken = cms.InputTag("L1TkGlbMuons",""),
   TkMuonToken = cms.InputTag("L1TkMuons",""),                                            

   tkTrackerJetToken = cms.InputTag("L1TrackerJets","L1TrackerJets"),                                            
   tkCaloJetToken = cms.InputTag("L1TkCaloJets","L1TkCaloJets"),
   tkMetToken = cms.InputTag("L1TrackerEtMiss","MET"),
   tkMhtTokens = cms.VInputTag( cms.InputTag("L1TrackerHTMiss5GeV","L1TrackerHTMiss"),cms.InputTag("L1TrackerHTMiss10GeV","L1TrackerHTMiss"),cms.InputTag("L1TrackerHTMiss","L1TrackerHTMiss"),cms.InputTag("L1TrackerHTMiss20GeV","L1TrackerHTMiss"),cms.InputTag("L1TrackerHTMiss30GeV","L1TrackerHTMiss")),

   ak4L1PF = cms.InputTag("ak4L1Puppi"),
 
   muonKalman = cms.InputTag("simKBmtfDigis","BMTF"),

   l1PFMet = cms.InputTag("l1MetPuppi"),

   zoPuppi = cms.InputTag("l1pfProducer","z0"),
   l1vertextdr = cms.InputTag("VertexProducer","l1vertextdr"),
   l1vertices = cms.InputTag("VertexProducer","l1vertices"),
   l1TkPrimaryVertex= cms.InputTag("L1TkPrimaryVertex",""),

   maxL1Extra = cms.uint32(20)
)

#### Gen level tree

genTree = cms.EDAnalyzer(
    "L1GenTreeProducer",
    genJetToken     = cms.untracked.InputTag("ak4GenJetsNoNu"),
    genMETTrueToken = cms.untracked.InputTag("genMetTrue"),
    genMETCaloToken     = cms.untracked.InputTag("genMetCalo"),
    genParticleToken = cms.untracked.InputTag("genParticles"),
    pileupInfoToken     = cms.untracked.InputTag("addPileupInfo")
)

runmenutree=cms.Path(l1PhaseIITree*genTree)


#### Additional collections that right now only the menu team is using (could be moved to L1TkObjectProducers_cff and SimL1Emulator_cff) 

from L1Trigger.L1TTrackMatch.L1TkElectronTrackProducer_cfi import L1TkElectrons
L1TkElectronsCrystal = L1TkElectrons.clone()
L1TkElectronsCrystal.L1EGammaInputTag = cms.InputTag("l1EGammaCrystalsProducer","L1EGammaCollectionBXVWithCuts") 
L1TkElectronsCrystal.IsoCut = cms.double(-0.1)

L1TkElectronsHGC=L1TkElectrons.clone()
L1TkElectronsHGC.L1EGammaInputTag = cms.InputTag("l1EGammaEEProducer","L1EGammaCollectionBXVWithCuts")
L1TkElectronsHGC.IsoCut = cms.double(-0.1)

from L1Trigger.L1TTrackMatch.L1TkEmParticleProducer_cfi import L1TkPhotons
L1TkPhotonsCrystal=L1TkPhotons.clone()
L1TkPhotonsCrystal.L1EGammaInputTag = cms.InputTag("l1EGammaCrystalsProducer","L1EGammaCollectionBXVWithCuts")
L1TkPhotonsCrystal.IsoCut = cms.double(-0.1)

L1TkPhotonsHGC=L1TkPhotons.clone()
L1TkPhotonsHGC.L1EGammaInputTag = cms.InputTag("l1EGammaEEProducer","L1EGammaCollectionBXVWithCuts")
L1TkPhotonsHGC.IsoCut = cms.double(-0.1)

L1TkElectronsLooseHGC = L1TkElectronsHGC.clone()
L1TkElectronsLooseHGC.TrackEGammaDeltaPhi = cms.vdouble(0.07, 0.0, 0.0)
L1TkElectronsLooseHGC.TrackEGammaDeltaR = cms.vdouble(0.12, 0.0, 0.0)
L1TkElectronsLooseHGC.TrackMinPt = cms.double( 3.0 )

L1TkElectronsLooseCrystal = L1TkElectronsCrystal.clone()
L1TkElectronsLooseCrystal.TrackEGammaDeltaPhi = cms.vdouble(0.07, 0.0, 0.0)
L1TkElectronsLooseCrystal.TrackEGammaDeltaR = cms.vdouble(0.12, 0.0, 0.0)
L1TkElectronsLooseCrystal.TrackMinPt = cms.double( 3.0 )

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


extraCollectionsMenuTree=cms.Path(L1TrackerHTMiss5GeV* L1TrackerHTMiss10GeV*L1TrackerHTMiss20GeV*L1TrackerHTMiss30GeV*L1TkElectronsCrystal*L1TkPhotonsCrystal*L1TkElectronsHGC*L1TkPhotonsHGC*L1TkElectronsLooseCrystal*L1TkElectronsLooseHGC*l1PhaseIITree*genTree)


