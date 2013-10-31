
import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
from RecoTauTag.RecoTau.PFRecoTauEnergyAlgorithmPlugin_cfi import pfTauEnergyAlgorithmPlugin

'''

Configuration for combinatoric PFTau producer plugins.

Note that this plugin produces many taus for each PFJet!
To be useful the output from this module must be cleaned
using an implementation of the RecoTauCleaner module.

Author: Evan K. Friis, UC Davis


'''

# N.B. for combinatoric taus that worst-case scaling
# is (maxTracks choose dmTracks) * (maxPiZeros choose dmPiZeros)
#
# So for decay mode 11 (3 tracks, 1 pizero), with 10 for both
#
# (10 choose 3) * (10 choose 1) = 1200!

# Configurations for the different decay modes

combinatoricDecayModeConfigs = cms.PSet(
    config1prong0pi0 = cms.PSet(
        # One prong no pizero mode
        nCharged = cms.uint32(1),
        nPiZeros = cms.uint32(0),
        maxTracks = cms.uint32(6),
        maxPiZeros = cms.uint32(0),
    ),
    config1prong1pi0 = cms.PSet(
        #One prong one pizero mode
        nCharged = cms.uint32(1),
        nPiZeros = cms.uint32(1),
        maxTracks = cms.uint32(6),
        maxPiZeros = cms.uint32(6),
    ),
    config1prong2pi0 = cms.PSet(
        #One prong two pizero mode
        nCharged = cms.uint32(1),
        nPiZeros = cms.uint32(2),
        maxTracks = cms.uint32(6),
        maxPiZeros = cms.uint32(5),
    ),
    config2prong0pi0 = cms.PSet(
        # Three prong no pizero mode (one of the tracks failed to get reconstructed)
        nCharged = cms.uint32(2),
        nPiZeros = cms.uint32(0),
        maxTracks = cms.uint32(6),
        maxPiZeros = cms.uint32(0),
    ),
    config2prong1pi0 = cms.PSet(
        # Three prong one pizero mode (one of the tracks failed to get reconstructed)
        nCharged = cms.uint32(2),
        nPiZeros = cms.uint32(1),
        maxTracks = cms.uint32(6),
        maxPiZeros = cms.uint32(3),
    ),
    config3prong0pi0 = cms.PSet(
        # Three prong no pizero mode
        nCharged = cms.uint32(3),
        nPiZeros = cms.uint32(0),
        maxTracks = cms.uint32(6),
        maxPiZeros = cms.uint32(0),
    )
)

_combinatoricTauConfig = cms.PSet(
    name = cms.string("combinatoric"),
    plugin = cms.string("RecoTauBuilderCombinatoricPlugin"),
    pfCandSrc = cms.InputTag("particleFlow"),
    isolationConeSize = cms.double(0.5),
    qualityCuts = PFTauQualityCuts,
    decayModes = cms.VPSet(
        combinatoricDecayModeConfigs.config1prong0pi0,
        combinatoricDecayModeConfigs.config1prong1pi0,
        combinatoricDecayModeConfigs.config1prong2pi0,
        combinatoricDecayModeConfigs.config2prong0pi0,
        combinatoricDecayModeConfigs.config2prong1pi0,
        combinatoricDecayModeConfigs.config3prong0pi0
    )
)

combinatoricRecoTaus = cms.EDProducer("RecoTauProducer",
    jetSrc = cms.InputTag("ak5PFJets"),
    jetRegionSrc = cms.InputTag("recoTauAK5PFJets08Region"),
    chargedHadronSrc = cms.InputTag('ak5PFJetsRecoTauChargedHadrons'),                                
    piZeroSrc = cms.InputTag("ak5PFJetsRecoTauPiZeros"),
    buildNullTaus = cms.bool(True),
    # Make maximum size from which to collect isolation cone objects, w.r.t to
    # the axis of the signal cone objects
    builders = cms.VPSet(
        _combinatoricTauConfig
    ),
    modifiers = cms.VPSet(
        cms.PSet(
            name = cms.string("sipt"),
            plugin = cms.string("RecoTauImpactParameterSignificancePlugin"),
            qualityCuts = PFTauQualityCuts,
        ),
        # Electron rejection
        cms.PSet(
            name = cms.string("elec_rej"),
            plugin = cms.string("RecoTauElectronRejectionPlugin"),
            #Electron rejection parameters
            ElectronPreIDProducer                = cms.InputTag("elecpreid"),
            EcalStripSumE_deltaPhiOverQ_minValue = cms.double(-0.1),
            EcalStripSumE_deltaPhiOverQ_maxValue = cms.double(0.5),
            EcalStripSumE_minClusEnergy          = cms.double(0.1),
            EcalStripSumE_deltaEta               = cms.double(0.03),
            ElecPreIDLeadTkMatch_maxDR           = cms.double(0.01),
            maximumForElectrionPreIDOutput       = cms.double(-0.1),
            DataType = cms.string("AOD"),
        ),
        # Tau energy reconstruction
        # (to avoid double-counting of energy carried by neutral PFCandidates
        #  in case PFRecoTauChargedHadrons are built from reco::Tracks)                                          
        cms.PSet(
            pfTauEnergyAlgorithmPlugin,
            name = cms.string("tau_en_reconstruction"),
            plugin = cms.string("PFRecoTauEnergyAlgorithmPlugin"),
            verbosity = cms.int32(0)                                              
        )
    )
)
