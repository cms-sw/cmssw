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
                                 
                                 ## Ecal Loose Id 
                                 l_EcalPhiWedgeEnergy = cms.double(10.),
                                 l_EcalPhiWedgeConstituents = cms.int32(6),
                                 l_EcalPhiWedgeToF = cms.double(-200.),  ### needs to be tuned when absolute timing in  EB/EE is understood w.r.t LHC
                                 l_EcalPhiWedgeConfidence = cms.double(.7),
                                 l_EcalShowerShapesRoundness = cms.double(.41),
                                 l_EcalShowerShapesAngle = cms.double(.51),
                                 l_EcalSuperClusterEnergy = cms.double(10.), # This  will be Et
                                 l_EcalSuperClusterSize = cms.int32(3),

                                 ## Ecal Tight Id
                                 t_EcalPhiWedgeEnergy = cms.double(20.),
                                 t_EcalPhiWedgeConstituents = cms.int32(8),
                                 t_EcalPhiWedgeToF = cms.double(-200.), ### needs to be tuned when absolute timing in  EB/EE is understood w.r.t LHC
                                 t_EcalPhiWedgeConfidence = cms.double(0.9),
                                 t_EcalShowerShapesRoundness = cms.double(.23),
                                 t_EcalShowerShapesAngle = cms.double(0.51),
                                 t_EcalSuperClusterEnergy = cms.double(10.), # This will be Et 
                                 t_EcalSuperClusterSize = cms.int32(3),

                                 ## Hcal Loose Id 
                                 l_HcalPhiWedgeEnergy = cms.double(20.),
                                 l_HcalPhiWedgeConstituents = cms.int32(6),
                                 l_HcalPhiWedgeToF = cms.double(-100.),  ### needs to be tuned when absolute timing in  HB/HE is understood w.r.t LHC
                                 l_HcalPhiWedgeConfidence = cms.double(0.7),

                                 ## Hcal Tight Id
                                 t_HcalPhiWedgeEnergy = cms.double(25.),
                                 t_HcalPhiWedgeConstituents = cms.int32(8),
                                 t_HcalPhiWedgeToF = cms.double(-100.), ### needs to be tuned when absolute timing in  HB/HE is understood w.r.t LHC
                                 t_HcalPhiWedgeConfidence = cms.double(0.9),

                                 # strips of problematic cells in HCAL min cut
                                 problematicStripMinLength = cms.int32(6)
                                 
                                 )

