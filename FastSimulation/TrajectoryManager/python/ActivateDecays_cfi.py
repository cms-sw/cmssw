import FWCore.ParameterSet.Config as cms

ActivateDecaysBlock = cms.PSet(
    ActivateDecays = cms.PSet(
        ActivateDecays = cms.bool(True),
        # Ahem, we initialize pythia with PYINIT, and this must be configurable...
        # (Temporary, until a better solution is found)
        comEnergy = cms.untracked.double(14000.0),
        # Maximum angle to associate a charged daughter to a charged mother
        # Mostly done to associate muons to decaying pions
        DistCut = cms.double(0.02)
    )
)

