import FWCore.ParameterSet.Config as cms

from Alignment.MuonAlignment.Scenarios_cff import *
MisalignedMuon = cms.ESProducer("MisalignedMuonESProducer",
    MuonNoMovementsScenario
)


