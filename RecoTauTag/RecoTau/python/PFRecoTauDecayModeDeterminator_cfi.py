import FWCore.ParameterSet.Config as cms
import copy

"""
        Defintions used in PFTauDecayModeDeterminator
        Purpose: Reconstruct the correct tau decay mode of reco::PFTaus
        Author: Evan K. Friis, UC Davis (friis@physics.ucdavis.edu)
"""

standardDecayModeParams = cms.PSet(
    maxPiZeroMass               = cms.double(0.2),  # Max mass of photon pairs that can be merged into a pi0
    refitTracks                 = cms.bool(False),  # Fit vertex for 3-prongs? (Not available on AOD data)
    mergeLowPtPhotonsFirst      = cms.bool(True),   # as opposed to highest pt first (only when mergeByBestMatch = false)
    mergeByBestMatch            = cms.bool(True),   # Compare each candidate pair of photons and merge the best one
    setMergedPi0Mass            = cms.bool(True),   # Set mass for merged photons?
    setChargedPionMass          = cms.bool(True),   # Set tracks mass to M_pi+?
    setPi0Mass                  = cms.bool(True),   # Set unmerged photons to M_pi0?
    filterPhotons               = cms.bool(True),   # Remove unmerged/merged gammas by the following two criteria:
    minPtFractionSinglePhotons  = cms.double(0.10), # Minimum pt fraction for unmerged photons to be included
    minPtFractionPiZeroes       = cms.double(0.15), # Minimum pt fraction for merged photons to be included
    maxPhotonsToMerge           = cms.uint32(2),    # Number of photons that can be put in a candidate pi0
    filterTwoProngs             = cms.bool(True),   # Filter two prongs
    minPtFractionForSecondProng = cms.double(0.1),  # second prong pt/lead track pt fraction when filterTwoProngs == True
    maxDistance                 = cms.double(0.01), # passed to vertex fitter when refitTracks is true
    maxNbrOfIterations          = cms.int32(10)     # passed to vertex fitter when refitTracks is true
)

pfTauDecayMode = cms.EDProducer("PFRecoTauDecayModeDeterminator",
      standardDecayModeParams,
      PFTauProducer = cms.InputTag("pfRecoTauProducer"),
)
pfTauDecayModeHighEfficiency = cms.EDProducer("PFRecoTauDecayModeDeterminator",
    standardDecayModeParams,
    PFTauProducer = cms.InputTag("pfRecoTauProducerHighEfficiency"),
)
pfTauDecayModeInsideOut = cms.EDProducer("PFRecoTauDecayModeDeterminator",
    standardDecayModeParams,
    PFTauProducer = cms.InputTag("pfRecoTauProducerInsideOut"),
)
