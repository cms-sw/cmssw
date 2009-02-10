import FWCore.ParameterSet.Config as cms

hbhereco = cms.EDFilter("HcalHitReconstructor",
                        correctionPhaseNS = cms.double(13.0),
                        digiLabel = cms.InputTag("hcalDigis"),
                        samplesToAdd = cms.int32(4),
                        Subdetector = cms.string('HBHE'),
                        firstSample = cms.int32(4),
                        correctForPhaseContainment = cms.bool(True),
                        correctForTimeslew = cms.bool(True),
                        
                        # Tags for calculating status flags
                        flagParameters= cms.PSet(
                            nominalPedestal=cms.double(3.0),  #fC
                            hitEnergyMinimum=cms.double(2.0), #GeV
                            hitMultiplicityThreshold=cms.int32(17),
                            pulseShapeParameterSets = cms.VPSet(
                              cms.PSet(pulseShapeParameters=cms.vdouble(-100.0, 20.0,-50.0,0.0,-15.0,0.0)),
                              cms.PSet(pulseShapeParameters=cms.vdouble( 100.0,2.0e3,-50.0,0.0,-15.0,0.05)),
                              cms.PSet(pulseShapeParameters=cms.vdouble( 2.0e3,1.0e6,-50.0,0.0, 85.0,0.0))
                              )
                            )
)
