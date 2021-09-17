import FWCore.ParameterSet.Config as cms

looseMuonCut = " (isGlobalMuon || isTrackerMuon) && isPFMuon"
looseIsoCut  = "(pfIsolationR04.sumChargedHadronPt + max(0., pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - 0.5 * pfIsolationR04.sumPUPt) ) / pt < 0.25"

tightMuonCut = " isGlobalMuon && isPFMuon && globalTrack.normalizedChi2 < 10. && globalTrack.hitPattern.numberOfValidMuonHits > 0 && " + \
    "numberOfMatchedStations > 1 && innerTrack.hitPattern.numberOfValidPixelHits > 0 && innerTrack.hitPattern.trackerLayersWithMeasurement > 5"
# CB PV cut!
tightIsoCut  = "(pfIsolationR04.sumChargedHadronPt + max(0., pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - 0.5 * pfIsolationR04.sumPUPt) ) / pt < 0.15"


EletightIsoCut  = "(pfIsolationVariables.sumChargedHadronPt + max(0., pfIsolationVariables.sumNeutralHadronEt + pfIsolationVariables.sumPhotonEt - 0.5 * pfIsolationVariables.sumPUPt) ) / pt < 0.1"
ElelooseIsoCut  = "(pfIsolationVariables.sumChargedHadronPt + max(0., pfIsolationVariables.sumNeutralHadronEt + pfIsolationVariables.sumPhotonEt - 0.5 * pfIsolationVariables.sumPUPt) ) / pt < 0.15"


looseElecCut = "((full5x5_sigmaIetaIeta < 0.011 && superCluster.isNonnull && superCluster.seed.isNonnull && (deltaEtaSuperClusterTrackAtVtx - superCluster.eta + superCluster.seed.eta) < 0.00477 && abs(deltaPhiSuperClusterTrackAtVtx) < 0.222 && hadronicOverEm < 0.298 && abs(1.0 - eSuperClusterOverP)*1.0/ecalEnergy < 0.241 && gsfTrack.hitPattern.numberOfHits('MISSING_INNER_HITS') <= 1 && abs(superCluster.eta) < 1.479) ||  (full5x5_sigmaIetaIeta() < 0.0314 && superCluster.isNonnull && superCluster.seed.isNonnull && (deltaEtaSuperClusterTrackAtVtx - superCluster.eta + superCluster.seed.eta) < 0.00868 && abs(deltaPhiSuperClusterTrackAtVtx) < 0.213 && hadronicOverEm < 0.101  && abs(1.0 - eSuperClusterOverP)*1.0/ecalEnergy < 0.14 && gsfTrack.hitPattern.numberOfHits('MISSING_INNER_HITS') <= 1 && abs(superCluster.eta) > 1.479))"

elecIPcut = "(abs(gsfTrack.d0)<0.05 & abs(gsfTrack.dz)<0.1 & abs(superCluster.eta) < 1.479)||(abs(gsfTrack.d0)<0.1 && abs(gsfTrack.dz)<0.2 && abs(superCluster.eta) > 1.479)"


tightElecCut = "((full5x5_sigmaIetaIeta < 0.00998 && superCluster.isNonnull && superCluster.seed.isNonnull && (deltaEtaSuperClusterTrackAtVtx - superCluster.eta + superCluster.seed.eta) < 0.00308 && abs(deltaPhiSuperClusterTrackAtVtx) < 0.0816 && hadronicOverEm < 0.0414 && abs(1.0 - eSuperClusterOverP)*1.0/ecalEnergy < 0.0129 && gsfTrack.hitPattern().numberOfLostHits('MISSING_INNER_HITS') <= 1 && abs(superCluster.eta) < 1.479) ||  (full5x5_sigmaIetaIeta() < 0.0292 && superCluster.isNonnull && superCluster.seed.isNonnull && (deltaEtaSuperClusterTrackAtVtx - superCluster.eta + superCluster.seed.eta) < 0.00605 && abs(deltaPhiSuperClusterTrackAtVtx) < 0.0394 && hadronicOverEm < 0.0641  && abs(1.0 - eSuperClusterOverP)*1.0/ecalEnergy < 0.0129 && gsfTrack.hitPattern().numberOfLostHits('MISSING_INNER_HITS') <= 1 && abs(superCluster.eta) > 1.479))"

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

singleTopMuonMediumDQM_miniAOD = DQMEDAnalyzer('SingleTopTChannelLeptonDQM_miniAOD',
  setup = cms.PSet(

    directory = cms.string("Physics/Top/SingleTopMuonMediumDQM_miniAOD/"),
    sources = cms.PSet(
      muons = cms.InputTag("slimmedMuons"),
      elecs = cms.InputTag("slimmedElectrons"),
      jets  = cms.InputTag("slimmedJets"),
      mets  = cms.VInputTag("slimmedMETs", "slimmedMETsNoHF", "slimmedMETsPuppi"),
      pvs   = cms.InputTag("offlineSlimmedPrimaryVertices")
    ),

    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),

    pvExtras = cms.PSet(
      select = cms.string("abs(z) < 24. & position.rho < 2. & ndof > 4 & !isFake")
    ),
    elecExtras = cms.PSet(
      select = cms.string(tightElecCut + "&& pt>20 & abs(eta)<2.5 & (abs(superCluster.eta) <= 1.4442 || abs(superCluster.eta) >= 1.5660)"),
      rho = cms.InputTag("fixedGridRhoFastjetAll"),
      #isolation = cms.string(ElelooseIsoCut),
    ),
                     
    muonExtras = cms.PSet(
      select    = cms.string(tightMuonCut + " && pt>20 & abs(eta)<2.4 && " + looseIsoCut),
      isolation = cms.string(looseIsoCut)
    ),
    jetExtras = cms.PSet(

      select = cms.string("pt>30 & abs(eta)<2.4"), # & neutralEmEnergyFraction >0.01 & chargedEmEnergyFraction>0.01"),
      jetBTaggers  = cms.PSet(
        trackCountingEff = cms.PSet(
          label = cms.InputTag("trackCountingHighEffBJetTags" ),
          workingPoint = cms.double(1.25)
        ),
        trackCountingPur = cms.PSet(
         label = cms.InputTag("trackCountingHighPurBJetTags" ),
          workingPoint = cms.double(3.41)
        ),
        secondaryVertex  = cms.PSet(
          label = cms.InputTag("simpleSecondaryVertexHighEffBJetTags"),
          workingPoint = cms.double(2.05)
        ),
        combinedSecondaryVertex  = cms.PSet(
          label = cms.InputTag("combinedSecondaryVertexBJetTags"),
          workingPoint = cms.double(0.898)
        )
     ),
   ),
   massExtras = cms.PSet(
     lowerEdge = cms.double( 70.),
     upperEdge = cms.double(110.)
   ),
  ),
  preselection = cms.PSet(
    vertex = cms.PSet(
      src    = cms.InputTag("offlineSlimmedPrimaryVertices"),#,
      select = cms.string("abs(z) < 24. & position.rho < 2. & ndof > 4 & !isFake")
    )
  ),

  selection = cms.VPSet(
    cms.PSet(
      label  = cms.string("muons:step0"),
      src    = cms.InputTag("slimmedMuons"),
      select = cms.string(tightMuonCut + " && pt>20 & abs(eta)<2.4 && " + looseIsoCut), #tightMuonCut +"&&"+ tightIsoCut + " && pt>20 & abs(eta)<2.1"), # CB what about iso? CD Added tightIso
      min    = cms.int32(1),
      #max    = cms.int32(1),
    ),
    cms.PSet(
      label  = cms.string("jets:step1"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.4 "),
      min = cms.int32(2),
    ),
    cms.PSet(
      label  = cms.string("met:step2"),
      src    = cms.InputTag("slimmedMETs"),
      select = cms.string("pt>30"),
      #min = cms.int32(2),
    ),
  )
)

singleTopElectronMediumDQM_miniAOD = DQMEDAnalyzer('SingleTopTChannelLeptonDQM_miniAOD',

  setup = cms.PSet(
 
    directory = cms.string("Physics/Top/SingleTopElectronMediumDQM_miniAOD/"),

    sources = cms.PSet(
      muons = cms.InputTag("slimmedMuons"),
      elecs = cms.InputTag("slimmedElectrons"),
      jets  = cms.InputTag("slimmedJets"),
      mets  = cms.VInputTag("slimmedMETs", "slimmedMETsNoHF", "slimmedMETsPuppi"),
      pvs   = cms.InputTag("offlineSlimmedPrimaryVertices")

    ),

    monitoring = cms.PSet(
      verbosity = cms.string("DEBUG")
    ),

    pvExtras = cms.PSet(
      select = cms.string("abs(z) < 24. & position.rho < 2. & ndof > 4 & !isFake")
    ),
    elecExtras = cms.PSet(
      select = cms.string(tightElecCut + " && pt>20 & abs(eta)<2.5 & (abs(superCluster.eta) <= 1.4442 || abs(superCluster.eta) >= 1.5660)"),
      #select     = cms.string(looseElecCut+ "&& pt>20 & abs(eta)<2.5 & (abs(superCluster.eta) <= 1.4442 || abs(superCluster.eta) >= 1.5660)"),
      rho = cms.InputTag("fixedGridRhoFastjetAll"),
      #isolation  = cms.string(ElelooseIsoCut),
    ),
    muonExtras = cms.PSet(
      select    = cms.string(tightMuonCut + " && pt>20 & abs(eta)<2.4 && " + looseIsoCut),
      isolation = cms.string(looseIsoCut)
    ),

    jetExtras = cms.PSet(

      select = cms.string("pt>30 & abs(eta)<2.4"),
      jetBTaggers  = cms.PSet(
        trackCountingEff = cms.PSet(
          label = cms.InputTag("trackCountingHighEffBJetTags" ),
          workingPoint = cms.double(1.25)
        ),
        trackCountingPur = cms.PSet(
          label = cms.InputTag("trackCountingHighPurBJetTags" ),
          workingPoint = cms.double(3.41)
        ),
        secondaryVertex  = cms.PSet(
          label = cms.InputTag("simpleSecondaryVertexHighEffBJetTags"),
          workingPoint = cms.double(2.05)
        ),
        combinedSecondaryVertex  = cms.PSet(
          label = cms.InputTag("combinedSecondaryVertexBJetTags"),
          workingPoint = cms.double(0.898)
        ),
      ),
    ),
    massExtras = cms.PSet(
      lowerEdge = cms.double( 70.),
      upperEdge = cms.double(110.)
    ),
               
  ),

  preselection = cms.PSet(
    vertex = cms.PSet(
      src    = cms.InputTag("offlineSlimmedPrimaryVertices"),#,
      select = cms.string("abs(z) < 24. & position.rho < 2. & ndof > 4 & !isFake")
    )
  ),

  selection = cms.VPSet(
    cms.PSet(
      label  = cms.string("elecs:step0"),
      src    = cms.InputTag("slimmedElectrons"),
      select = cms.string("pt>20 & abs(eta)<2.5 & (abs(superCluster.eta) <= 1.4442 || abs(superCluster.eta) >= 1.5660) &&" + tightElecCut),
      min    = cms.int32(1),
      max    = cms.int32(1),
    ),
    cms.PSet(
      label  = cms.string("jets:step1"),
      src    = cms.InputTag("slimmedJets"),
      select = cms.string("pt>30 & abs(eta)<2.4 "),
      min = cms.int32(2),
    ),
    cms.PSet(
      label  = cms.string("met:step2"),
      src    = cms.InputTag("slimmedMETs"),
      select = cms.string("pt>30"),
      #min = cms.int32(2),
    ),
  )
)
