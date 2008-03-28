import FWCore.ParameterSet.Config as cms

from RecoBTag.MCTools.mcJetFlavour_cff import *
btagCalibPath = cms.Path(mcAlgoJetFlavour+mcPhysJetFlavour)

