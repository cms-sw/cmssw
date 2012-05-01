import FWCore.ParameterSet.Config as cms

ActivateDecaysBlock = cms.PSet(
    ActivateDecays = cms.PSet(
        ActivateDecays = cms.bool(True),
        # External program used to handle decays: available options are pythia6 and pythia8
        Decayer = cms.string('pythia6'), 
        # Maximum angle to associate a charged daughter to a charged mother
        # Mostly done to associate muons to decaying pions
        DistCut = cms.double(0.02)
    )
)

