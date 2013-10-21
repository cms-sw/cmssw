import FWCore.ParameterSet.Config as cms

TauolaDefaultInputCards = cms.PSet(
   InputCards = cms.PSet
   (
      pjak1 = cms.int32(0),
      pjak2 = cms.int32(0),
      mdtau = cms.int32(0)
   ) 
)

TauolaNoPolar = cms.PSet(
    UseTauolaPolarization = cms.bool(False)
)

TauolaPolar = cms.PSet(
    UseTauolaPolarization = cms.bool(True)
)


