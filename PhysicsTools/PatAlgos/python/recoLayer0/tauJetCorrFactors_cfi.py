import FWCore.ParameterSet.Config as cms

from RecoTauTag.TauTagTools.tauDecayModes_cfi import *

# module to produce tau-jet energy correction factors
patTauJetCorrFactors = cms.EDProducer("TauJetCorrFactorsProducer",
    # input collection of jets
    src = cms.InputTag('hpsPFTauProducer'),
    # mapping of tau decay modes to payloads:
    # for reco::PFTaus, the decay modes are defined in DataFormats/TauReco/interface/PFTau.h ;
    # 'other' is taken for all reconstructed decay modes not explicitely specified
    # for reco::CaloTaus, for which no decay mode reconstruction has been implemented yet,
    # 'other' is taken for all tau-jet candidates
    parameters = cms.VPSet(
        cms.PSet(
            payload    = cms.string('AK5tauHPSlooseCombDBcorrOneProng0Pi0'),
            decayModes = cms.vstring('%i' % tauToOneProng0PiZero)
        ),
        cms.PSet(
            payload    = cms.string('AK5tauHPSlooseCombDBcorrOneProng1Pi0'),
            decayModes = cms.vstring('%i' % tauToOneProng1PiZero)
        ),
        cms.PSet(
            payload    = cms.string('AK5tauHPSlooseCombDBcorrOneProng2Pi0'),
            decayModes = cms.vstring('%i' % tauToOneProng2PiZero)
        ),
        cms.PSet(
            payload    = cms.string('AK5tauHPSlooseCombDBcorrThreeProng0Pi0'),
            decayModes = cms.vstring('%i' % tauToThreeProng0PiZero)
        ),
        cms.PSet(
            payload    = cms.string('AK5tauHPSlooseCombDBcorr'),
            decayModes = cms.vstring('*')
        )
    ),
    # correction levels
    levels = cms.vstring(
        # tags for the individual jet corrections;
        # when not available the string should be set to 'none'
        'L2Relative', 'L3Absolute'
    )
)
