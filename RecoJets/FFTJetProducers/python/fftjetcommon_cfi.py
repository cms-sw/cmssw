import math
import FWCore.ParameterSet.Config as cms
import RecoJets.JetProducers.AnomalousCellParameters_cfi as anom

# Common definitions for FFTJet interface modules

# Some useful constants
fftjet_large_int = pow(2,31) - 1

# Global phi-to-eta bandwidth ratio
fftjet_phi_to_eta_bw_ratio = 1.0

# Are we including the complete event at the lowest scale
# of the clustering tree?
fftjet_insert_complete_event = False
fftjet_complete_event_scale = 0.05

# The standard eta range for various algos
fftjet_standard_eta_range = 5.2

# Provide several feasible energy discretization grid configurations.
# The width of eta bins is 0.087. We want to choose the binning
# so that FFT is efficient. At the same time, we have to prevent
# wrap-around energy leakage during convolutions from one eta side
# to another. Note that the CMS calorimeter extends up to eta of 5.191.
fftjet_grid_256_72 = cms.PSet(
    nEtaBins = cms.uint32(256),
    etaMin = cms.double(-11.136),
    etaMax = cms.double(11.136),
    nPhiBins = cms.uint32(72),
    phiBin0Edge = cms.double(0.0),
    title = cms.untracked.string("256 x 72")
)

fftjet_grid_192_72 = cms.PSet(
    nEtaBins = cms.uint32(192),
    etaMin = cms.double(-8.352),
    etaMax = cms.double(8.352),
    nPhiBins = cms.uint32(72),
    phiBin0Edge = cms.double(0.0),
    title = cms.untracked.string("192 x 72")
)

fftjet_grid_144_72 = cms.PSet(
    nEtaBins = cms.uint32(144),
    etaMin = cms.double(-6.264),
    etaMax = cms.double(6.264),
    nPhiBins = cms.uint32(72),
    phiBin0Edge = cms.double(0.0),
    title = cms.untracked.string("144 x 72")
)

fftjet_grid_128_72 = cms.PSet(
    nEtaBins = cms.uint32(128),
    etaMin = cms.double(-5.568),
    etaMax = cms.double(5.568),
    nPhiBins = cms.uint32(72),
    phiBin0Edge = cms.double(0.0),
    title = cms.untracked.string("128 x 72")
)

fftjet_grid_256_128 = cms.PSet(
    nEtaBins = cms.uint32(256),
    etaMin = cms.double(-2.0*math.pi),
    etaMax = cms.double(2.0*math.pi),
    nPhiBins = cms.uint32(128),
    phiBin0Edge = cms.double(0.0),
    title = cms.untracked.string("256 x 128")
)

#
# Definitions for anomalous towers
# 
fftjet_anomalous_tower_default = anom.AnomalousCellParameters

fftjet_anomalous_tower_allpass = cms.PSet(
    maxBadEcalCells = cms.uint32(fftjet_large_int),
    maxRecoveredEcalCells = cms.uint32(fftjet_large_int),
    maxProblematicEcalCells = cms.uint32(fftjet_large_int),
    maxBadHcalCells = cms.uint32(fftjet_large_int),
    maxRecoveredHcalCells = cms.uint32(fftjet_large_int),
    maxProblematicHcalCells = cms.uint32(fftjet_large_int)
)

#
# Peak selectors
# 
fftjet_peak_selector_allpass = cms.PSet(
    Class = cms.string("AllPeaksPass")
)

#
# 50 scales (49 intervals) from 0.087 to 0.6 in log space correspond
# to the pattern recognition kernel bandwidth increase of 4.0% per scale.
# This set of scales is useful for generic jet reconstruction with
# variable jet size and for multiresolution studies.
#
fftjet_patreco_scales_50 = cms.PSet(
    Class = cms.string("EquidistantInLogSpace"),
    minScale = cms.double(0.087),
    maxScale = cms.double(0.6),
    nScales = cms.uint32(50)
)

#
# Various sets of scales with 3 values (and 2 intervals).
# Central scale is to be used for single-resolution jet
# reconstruction. Using 3 scales instead of 1 allows for
# determination of varios "speed" quantities for peaks.
#
fftjet_patreco_scales_3_at_010 = cms.PSet(
    Class = cms.string("EquidistantInLogSpace"),
    minScale = cms.double(0.10/1.04),
    maxScale = cms.double(0.10*1.04),
    nScales = cms.uint32(3)
)
fftjet_patreco_scales_3_at_015 = cms.PSet(
    Class = cms.string("EquidistantInLogSpace"),
    minScale = cms.double(0.15/1.04),
    maxScale = cms.double(0.15*1.04),
    nScales = cms.uint32(3)
)
fftjet_patreco_scales_3_at_017 = cms.PSet(
    Class = cms.string("EquidistantInLogSpace"),
    minScale = cms.double(0.17/1.04),
    maxScale = cms.double(0.17*1.04),
    nScales = cms.uint32(3)
)
fftjet_patreco_scales_3_at_020 = cms.PSet(
    Class = cms.string("EquidistantInLogSpace"),
    minScale = cms.double(0.20/1.04),
    maxScale = cms.double(0.20*1.04),
    nScales = cms.uint32(3)
)
fftjet_patreco_scales_3_at_025 = cms.PSet(
    Class = cms.string("EquidistantInLogSpace"),
    minScale = cms.double(0.25/1.04),
    maxScale = cms.double(0.25*1.04),
    nScales = cms.uint32(3)
)
fftjet_patreco_scales_3_at_050 = cms.PSet(
    Class = cms.string("EquidistantInLogSpace"),
    minScale = cms.double(0.50/1.04),
    maxScale = cms.double(0.50*1.04),
    nScales = cms.uint32(3)
)

#
# Here, the distance calculator for the tree is a simple eta-phi
# distance with fixed bandwidth values in eta and phi. However,
# if the "etaDependentScaleFactors" are given in the module
# configuration, it makes a lot of sense to use eta-dependent
# eta-to-phi bandwidth ratio.
#
fftjet_fixed_bandwidth_distance = cms.PSet(
    Class = cms.string("PeakEtaPhiDistance"),
    etaToPhiBandwidthRatio = cms.double(1.0/fftjet_phi_to_eta_bw_ratio)
)

#
# A placeholder for the tree distance calculator with eta-dependent
# eta-to-phi bandwidth ratio (must be modified for meaningful use).
# Inside the interpolator, bandwidth ratio points are placed at the
# cell centers.
#
fftjet_variable_bandwidth_distance = cms.PSet(
    Class = cms.string("PeakEtaDependentDistance"),
    Interpolator = cms.PSet(
        xmin = cms.double(-5.2),
        xmax = cms.double(5.2),
        flow = cms.double(1.0),
        fhigh = cms.double(1.0),
        data = cms.vdouble(1.0, 1.0)
    )
)

#
# Various jet membership functions
#
fftjet_jet_membership_cone = cms.PSet(
    Class = cms.string("Linear2d"),
    sx = cms.double(math.sqrt(1.0/fftjet_phi_to_eta_bw_ratio)),
    sy = cms.double(math.sqrt(fftjet_phi_to_eta_bw_ratio)),
    scalePower = cms.int32(1),
    kernelParameters = cms.vdouble()
)

#
# Background/noise membership functions
#
fftjet_noise_membership_smallconst = cms.PSet(
    Class = cms.string("GaussianNoiseMembershipFcn"),
    minWeight = cms.double(1.0e-8),
    prior = cms.double(0.0)
)

#
# Distance between jets for convergence determination
#
fftjet_convergence_jet_distance = cms.PSet(
    Class = cms.string("JetConvergenceDistance"),
    etaToPhiBandwidthRatio = cms.double(1.0/fftjet_phi_to_eta_bw_ratio),
    relativePtBandwidth = cms.double(1.0)
)

#
# Various peak functors
#
fftjet_peakfunctor_const_zero = cms.PSet(
    Class = cms.string("ConstDouble"),
    value = cms.double(0.0)
)

fftjet_peakfunctor_const_one = cms.PSet(
    Class = cms.string("ConstDouble"),
    value = cms.double(1.0)
)
