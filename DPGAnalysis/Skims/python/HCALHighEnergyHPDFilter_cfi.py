import FWCore.ParameterSet.Config as cms

HCALHighEnergyHPDFilter = cms.EDFilter("HLTHPDFilter",
                                       inputTag = cms.InputTag("hbhereco"),
                                       energy = cms.double(-99.0),
                                       hpdSpikeEnergy = cms.double(10.0),
                                       hpdSpikeIsolationEnergy = cms.double(1.0),
                                       rbxSpikeEnergy = cms.double(50.0),
                                       rbxSpikeUnbalance = cms.double(0.2)
                                       )
