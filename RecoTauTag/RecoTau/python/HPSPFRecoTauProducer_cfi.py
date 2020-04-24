import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
from RecoTauTag.RecoTau.PFRecoTauProducer_cfi import pfRecoTauProducer

hpsPFRecoTauProducer = pfRecoTauProducer.clone(
      #Standard Input
      PFTauTagInfoProducer = cms.InputTag("pfRecoTauTagInfoProducer"),
      JetPtMin             = cms.double(0.0),
      ElectronPreIDProducer                = cms.InputTag("elecpreid"),
      PVProducer      = PFTauQualityCuts.primaryVertexSrc,
      smearedPVsigmaY = cms.double(0.0015),
      smearedPVsigmaX = cms.double(0.0015),
      smearedPVsigmaZ = cms.double(0.005),

      Algorithm       = cms.string("HPS"),
      #HPS Specific

      #Setup the merger
      emMergingAlgorithm           = cms.string("StripBased"), #Strip Algorithm
      stripCandidatesPdgIds        = cms.vint32(22,11),        #Clusterize photons and electrons
      stripEtaAssociationDistance  = cms.double(0.05),         #Eta Association for the strips
      stripPhiAssociationDistance  = cms.double(0.2),          #Phi Association for the strips
      stripPtThreshold             = cms.double(1.0),          # Strip Pt Threshold

      candOverlapCriterion   = cms.string("Isolation"),        #Overlap filter double decay modes by isolation
      #Select Decay Modes
      doOneProng             = cms.bool(True),
      doOneProngStrip        = cms.bool(True),
      doOneProngTwoStrips    = cms.bool(True),
      doThreeProng           = cms.bool(True),

      #Minimum Pt for the tau
      tauPtThreshold         = cms.double(15.),
      #Leading Pion Threshold
      leadPionThreshold      = cms.double(1.0),


      #isolation cone definitions
      chargeHadrIsolationConeSize = cms.double(0.5),
      gammaIsolationConeSize      = cms.double(0.5),
      neutrHadrIsolationConeSize  = cms.double(0.5),

      # If it is true the algorithm will calculate the signal cone
      # and put in the isolation items that are outside it
      #If it is False it will associate to the isolation
      #any non signal constituent
      useIsolationAnnulus = cms.bool(False),
      #Mass Windows
      oneProngStripMassWindow = cms.vdouble(0.3,1.3),
      oneProngTwoStripsMassWindow = cms.vdouble(0.4,1.2),
      oneProngTwoStripsPi0MassWindow = cms.vdouble(0.05,0.2),
      threeProngMassWindow = cms.vdouble(0.8,1.5),
      #Matching cone for the taus
      matchingCone= cms.double(0.1),
      #Add converted electrons to the Tau LV
      coneMetric        = cms.string("DR"),
      coneSizeFormula   = cms.string("2.8/ET"),
      minimumSignalCone = cms.double(0.05),
      maximumSignalCone = cms.double(0.1)
)
