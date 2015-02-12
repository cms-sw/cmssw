#
# \version $Id: SinglePhotonAnalyzer_cfi.py,v 1.22 2011/07/20 19:52:57 kimy Exp $
# track

import FWCore.ParameterSet.Config as cms

from RecoHI.HiEgammaAlgos.HiIsolationCommonParameters_cff import *

singlePhotonAnalyzer = cms.EDAnalyzer("SinglePhotonAnalyzer",
                                      verbose                   = cms.untracked.bool(False),
                                      FillMCNTuple              = cms.untracked.bool(True),
                                      isMC_      = cms.untracked.bool(False),
                                      DoL1Objects               = cms.untracked.bool(False),
                                      StorePhysVectors          = cms.untracked.bool(False),         
                                      OutputFile                = cms.string('test.root'),
                                      HepMCProducer             = cms.InputTag("generator"),
                                      GenParticleProducer       = cms.InputTag("hiGenParticles"),
                                      GenEventScale             = cms.InputTag("hiSignal"),
                                      PhotonProducer            = cms.InputTag("selectedPatPhotons"),
                                      TrackProducer             = isolationInputParameters.track,
                                      JetProducer               = cms.InputTag("icPu5patJets"),
                                      METProducer               = cms.InputTag("layer1METs"),
                                      VertexProducer            = cms.InputTag("hiSelectedVertex"),
                                      BeamSpotProducer          = cms.InputTag("offlineBeamSpot" ),
                                      ebReducedRecHitCollection = cms.InputTag("ecalRecHit:EcalRecHitsEB"),
                                      eeReducedRecHitCollection = cms.InputTag("ecalRecHit:EcalRecHitsEE"),
                                      basicClusterBarrel        = cms.InputTag("islandBasicClusters:islandBarrelBasicClusters"),
                                      basicClusterEndcap        = cms.InputTag("islandBasicClusters:islandEndcapBasicClusters"),
                                      compPhotonProducer   = cms.InputTag("complePhoton"),
                                      hbhe                      = cms.InputTag("hbhereco"),
                                      hf                      = cms.InputTag("hfreco"),
                                      ho                      = cms.InputTag("horeco"),
                                      
                                      caloTowers                = cms.InputTag("towerMaker"),                                     
                                      HltTriggerResults         = cms.InputTag("TriggerResults::HLT"),
                                      L1gtReadout               = cms.InputTag("gtDigis"),
                                      L1IsolTag                 = cms.InputTag("l1extraParticles","Isolated"),
                                      L1NonIsolTag              = cms.InputTag("l1extraParticles","NonIsolated"),
                                      # The results of these trigger paths will get stored in the TTree by their name
                                      TriggerPathsToStore       = cms.vstring("HLT_MinBias", "HLT_MinBiasBSC", "HLT_MinBiasBSC_NoBPTX", "HLT_MinBiasBSC_OR",
                                                                              "HLT_L1SingleEG5",
                                                                              "HLT_L1SingleEG8",
                                                                              "HLT_Photon10_L1R",
                                                                              "HLT_Photon15_L1R",
                                                                              "HLT_Photon20_L1R"),
                                      GammaPtMin                = cms.untracked.double(15),
                                      GammaEtaMax               = cms.untracked.double(3.0),
                                      McPtMin                   = cms.untracked.double(10),
                                      McEtaMax                  = cms.untracked.double(3.5),
                                      EcalBarrelMaxEta          = cms.untracked.double(1.45),
                                      EcalEndcapMinEta          = cms.untracked.double(1.55),
                                      JetPtMin                  = cms.untracked.double(20),
                                      TrackPtMin                = cms.untracked.double(1.5),   # used for doStoreTrack
                                      TrackEtaMax               = cms.untracked.double(1.85),  # used for doStoreTrack
                                      etCutGenMatch             = cms.untracked.double(10),
                                      etaCutGenMatch            = cms.untracked.double(3.5),
                                      doStoreGeneral            = cms.untracked.bool(True),
                                      doStoreCentrality         = cms.untracked.bool(True),
                                      doStoreL1Trigger          = cms.untracked.bool(True),
                                      doStoreHLT                = cms.untracked.bool(True),
                                      doStoreHF                 = cms.untracked.bool(True),
                                      doStoreVertex             = cms.untracked.bool(True),
                                      doStoreMET                = cms.untracked.bool(False),
                                      doStoreJets               = cms.untracked.bool(False),
                                      doStoreCompCone           = cms.untracked.bool(True),
                                      doStoreConversions        = cms.untracked.bool(False),
                                      doStoreTracks             = cms.untracked.bool(True),
                                      gsfElectronCollection = cms.untracked.InputTag("gsfElectrons"),
                                      hiEvtPlane_               = cms.InputTag("hiEvtPlane","recoLevel"),
                                      pfCandidateLabel            =cms.InputTag("particleFlowTmp"),
                                      voronoiBkg                   = cms.InputTag("voronoiBackgroundPF"),
                                      towerCandidateLabel               = cms.InputTag("towerMaker"),
                                      towerVoronoiBkg                   = cms.InputTag("voronoiBackgroundCalo")
                                      )


