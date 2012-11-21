# FFTJet pile-up processor module configuration for CaloJets

import FWCore.ParameterSet.Config as cms

from RecoJets.FFTJetProducers.fftjetcommon_cfi import *
import RecoJets.FFTJetProducers.pileup_shape_Summer11_Calo_v1_cfi as calo_ps

# A good ratio is about 1.1 for PFJets, could be larger for CaloJets
fftjet_pileup_phi_to_eta_ratio = 1.1

# Note that for the grid below we do not really care that
# convolution results will wrap around in eta
fftjet_pileup_grid_calo = cms.PSet(
    nEtaBins = cms.uint32(calo_ps.fftjet_pileup_eta_bins),
    etaMin = cms.double(-calo_ps.fftjet_pileup_eta_max),
    etaMax = cms.double(calo_ps.fftjet_pileup_eta_max),
    nPhiBins = cms.uint32(calo_ps.fftjet_pileup_phi_bins),
    phiBin0Edge = cms.double(0.0),
    title = cms.untracked.string("FFTJet Pileup Grid")
)

fftjetPileupProcessorCalo = cms.EDProducer(
    "FFTJetPileupProcessor",
    #
    # The main eta and phi scale factors for the filters
    kernelEtaScale = cms.double(1.0),
    kernelPhiScale = cms.double(fftjet_pileup_phi_to_eta_ratio),
    #
    # Label for the produced objects
    outputLabel = cms.string("FFTJetPileupCalo"),
    #
    # Label for the input collection of Candidate objects
    src = cms.InputTag("towerMaker"),
    #
    # Label for the jets. Vertex correction may be done for "CaloJet" only.
    jetType = cms.string("CaloJet"),
    #
    # Perform vertex correction?
    doPVCorrection = cms.bool(False),
    #
    # Label for the input collection of vertex objects. Meaningful
    # only when "doPVCorrection" is True
    srcPVs = cms.InputTag("offlinePrimaryVertices"),
    #
    # Eta-dependent magnitude factors for the data (applied before filtering)
    etaDependentMagnutideFactors = cms.vdouble(),
    #
    # Eta-dependent magnitude factors for the data (applied after filtering)
    etaFlatteningFactors = cms.vdouble(calo_ps.fftjet_pileup_magnitude_factors),
    #
    # Configuration for the energy discretization grid
    GridConfiguration = fftjet_pileup_grid_calo,
    #
    # Convolution range
    convolverMinBin = cms.uint32(calo_ps.fftjet_pileup_min_eta_bin),
    convolverMaxBin = cms.uint32(calo_ps.fftjet_pileup_max_eta_bin),
    #
    # Conversion factor from the Et sum to the Et density
    pileupEtaPhiArea = cms.double(calo_ps.fftjet_pileup_eta_phi_area),
    #
    # The set of scales used by the filters. Here, just one scale
    # is configured
    nScales = cms.uint32(1),
    minScale = cms.double(calo_ps.fftjet_pileup_bandwidth),
    maxScale = cms.double(calo_ps.fftjet_pileup_bandwidth),
    #
    # The number of percentile points to use
    nPercentiles = cms.uint32(51),
    #
    # Files for mixing in external grids
    externalGridFiles = cms.vstring(),
    #
    # Energy cutoff for external grids (removes some crazy grids)
    externalGridMaxEnergy = cms.double(20000.0),
    #
    # Anomalous calo tower definition (comes from JetProducers default)
    anomalous = fftjet_anomalous_tower_default,
    #
    # Some parameters inherited from FFTJetInterface which no longer
    # play any role
    insertCompleteEvent = cms.bool(fftjet_insert_complete_event),
    completeEventScale = cms.double(fftjet_complete_event_scale),
    #
    # Parameters related to accessing the table of flattening factors from DB
    flatteningTableRecord = cms.string("flatteningTableRecord"),
    flatteningTableName = cms.string("flatteningTableName"),
    flatteningTableCategory = cms.string("flatteningTableCategory"),
    loadFlatteningFactorsFromDB = cms.bool(False)
)
