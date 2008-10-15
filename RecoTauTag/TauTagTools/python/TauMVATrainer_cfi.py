import FWCore.ParameterSet.Config as cms

standardHighEffDecayModeSetupSignal = cms.PSet(
     truthMatchSource = cms.InputTag("matchMCTausHighEfficiency"),
     decayModeAssociationSource = cms.InputTag("pfTauDecayModeHighEfficiency")
)

standardInsideOutDecayModeSetupSignal = cms.PSet(
     truthMatchSource = cms.InputTag("matchMCTausInsideOut"),
     decayModeAssociationSource = cms.InputTag("pfTauDecayModeInsideOut")
)

standardHighEffDecayModeSetupBackground = cms.PSet(
     truthMatchSource = cms.InputTag("matchMCQCDHighEfficiency"),
     decayModeAssociationSource = cms.InputTag("pfTauDecayModeHighEfficiency")
)

standardInsideOutDecayModeSetupBackground = cms.PSet(
     truthMatchSource = cms.InputTag("matchMCQCDInsideOut"),
     decayModeAssociationSource = cms.InputTag("pfTauDecayModeInsideOut")
)

tauMVATrainerSignal = cms.EDProducer("TauMVATrainer",
    outputRootFileName = cms.string('output_signal.root'),
    #Inside-Out currently broken
    #matchingSources = cms.VPSet(standardHighEffDecayModeSetupSignal, standardInsideOutDecayModeSetupSignal),
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
    #Inside-Out currently broken
    #matchingSources = cms.VPSet(standardHighEffDecayModeSetupBackground, standardInsideOutDecayModeSetupBackground),
    matchingSources = cms.VPSet(standardHighEffDecayModeSetupBackground),
    #any object with a multiplicity larger than the given two parameters is
    #automatically marked as "__PREFAIL__" in the output tree
    maxPiZeroes = cms.uint32(4),
    maxTracks = cms.uint32(3),
    mcTruthSource = cms.InputTag("makeMCQCDTauDecayModes"),
    iAmSignal = cms.bool(False)
)


