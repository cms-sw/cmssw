import FWCore.ParameterSet.Config as cms

# Designed to disable a bug affecting long lived slepton decays in HepMC-G4 interface
fixLongLivedSleptonSim = cms.Modifier()
