import FWCore.ParameterSet.Config as cms
# File: BeamHaloSummary_cfi.py
# Original Author: R. Remington, The University of Florida
# Description: Module to build BeamHaloSummary Object and put into the event
# Date: Oct. 15, 2009

BeamHaloSummary = cms.EDProducer("BeamHaloSummaryProducer",
                             CSCHaloDataLabel = cms.InputTag("CSCHaloData"),
                             EcalHaloDataLabel = cms.InputTag("EcalHaloData"),
                             HcalHaloDataLabel = cms.InputTag("HcalHaloData"),
                             GlobalHaloDataLabel = cms.InputTag("GlobalHaloData"),

                             l_EcalPhiWedgeEnergy = cms.double(10.),
                             l_EcalPhiWedgeConstituents = cms.int32(6),
                             l_EcalPhiWedgeToF = cms.double(200.),
                             l_EcalPhiWedgeConfidence = cms.double(.7),
                             l_EcalShowerShapesRoundness = cms.double(.75),
                             l_EcalShowerShapesAngle = cms.double(1.2),

                             t_EcalPhiWedgeEnergy = cms.double(20.),
                             t_EcalPhiWedgeConstituents = cms.int32(8),
                             t_EcalPhiWedgeToF = cms.double(200.),
                             t_EcalPhiWedgeConfidence = cms.double(0.8),
                             t_EcalShowerShapesRoundness = cms.double(.5),
                             t_EcalShowerShapesAngle = cms.double(0.8), 

                             l_HcalPhiWedgeEnergy = cms.double(20.),
                             l_HcalPhiWedgeConstituents = cms.int32(6),
                             l_HcalPhiWedgeToF = cms.double(-25.),
                             l_HcalPhiWedgeConfidence = cms.double(0.7),

                             t_HcalPhiWedgeEnergy = cms.double(25.),
                             t_HcalPhiWedgeConstituents = cms.int32(8),
                             t_HcalPhiWedgeToF = cms.double(-30.),
                             t_HcalPhiWedgeConfidence = cms.double(0.8)
                             
                             )

