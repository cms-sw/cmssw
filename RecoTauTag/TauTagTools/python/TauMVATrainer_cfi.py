import FWCore.ParameterSet.Config as cms

standardHighEffDecayModeSetupSignal = cms.PSet(
     truthMatchSource = cms.InputTag("matchMCTausShrinkingCone"),
     decayModeAssociationSource = cms.InputTag("shrinkingConePFTauDecayModeProducer")
)

standardHighEffDecayModeSetupBackground = cms.PSet(
     truthMatchSource = cms.InputTag("matchMCQCDShrinkingCone"),
     decayModeAssociationSource = cms.InputTag("shrinkingConePFTauDecayModeProducer")
)

#### Future collections #####
standardInsideOutDecayModeSetupSignal = cms.PSet(
     truthMatchSource = cms.InputTag("matchMCTausInsideOut"),
     decayModeAssociationSource = cms.InputTag("pfTauDecayModeInsideOut")
)


standardInsideOutDecayModeSetupBackground = cms.PSet(
     truthMatchSource = cms.InputTag("matchMCQCDInsideOut"),
     decayModeAssociationSource = cms.InputTag("pfTauDecayModeInsideOut")
)

"""
#### Define the trainers for signal & background ####
You can add additional algorithms in the match sources, if you wish to compare performance.
"""
tauMVATrainerSignal = cms.EDProducer("TauMVATrainer",
    outputRootFileName = cms.string('output_signal.root'),
    matchingSources = cms.VPSet(standardHighEffDecayModeSetupSignal),
    #any object with a multiplicity larger than the given two parameters is
    #automatically marked as "__PREFAIL__" in the output tree
    maxPiZeroes = cms.uint32(4),
    maxTracks = cms.uint32(3),
    mcTruthSource = cms.InputTag("makeMCTauDecayModes"),
    iAmSignal = cms.bool(True)
)

tauMVATrainerBackground = cms.EDProducer("TauMVATrainer",
    outputRootFileName = cms.string('output_qcd.root'),
    matchingSources = cms.VPSet(standardHighEffDecayModeSetupBackground),                
    #any object with a multiplicity larger than the given two parameters is
    #automatically marked as "__PREFAIL__" in the output tree
    maxPiZeroes = cms.uint32(4),
    maxTracks = cms.uint32(3),
    mcTruthSource = cms.InputTag("makeMCQCDTauDecayModes"),
    iAmSignal = cms.bool(False)
)


