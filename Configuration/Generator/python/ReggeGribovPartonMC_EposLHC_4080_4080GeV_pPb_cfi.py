import FWCore.ParameterSet.Config as cms

generator = cms.EDFilter("ReggeGribovPartonMCGeneratorFilter",

                          bmin = cms.double(0), #impact parameter min in fm
                          bmax = cms.double(10000),#impact parameter max in fm
                          paramFileName = cms.untracked.string("Configuration/Generator/data/ReggeGribovPartonMC.param"), #file with more parameters specific to crmc interface
                          skipNuclFrag = cms.bool(True), #in HI collisions nuclear fragments with pt=0 can be in the hep event. to skip those activate this option
                          beammomentum = cms.double(4080),
                          targetmomentum = cms.double(-4080),
                          beamid = cms.int32(208),
                          targetid = cms.int32(1),
                          model = cms.int32(0),
                          )
