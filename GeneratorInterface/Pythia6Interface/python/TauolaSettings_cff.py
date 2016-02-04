import FWCore.ParameterSet.Config as cms

from GeneratorInterface.ExternalDecays.TauolaSettings_cff import *

TauolaDefaultInputForSource = cms.PSet(
    InputCards = cms.vstring('TAUOLA = 0 0 0   ! TAUOLA ')
)


#TauolaNoPolar = cms.PSet(
#    UseTauolaPolarization = cms.bool(False)
#)
#TauolaPolar = cms.PSet(
#    UseTauolaPolarization = cms.bool(True)
#)


