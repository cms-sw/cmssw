import FWCore.ParameterSet.Config as cms
# File: BeamHaloSummary_cfi.py
# Original Author: R. Remington, The University of Florida
# Description: Module to build BeamHaloSummary Object and put into the event
# Date: Oct. 15, 2009

halosummary = cms.EDProducer("BeamHaloSummaryProducer",
                             CSCHaloDataLabel = cms.InputTag("CSCHaloData"),
                             EcalHaloDataLabel = cms.InputTag("EcalHaloData"),
                             HcalHaloDataLabel = cms.InputTag("HcalHaloData"),
                             GlobalHaloDataLabel = cms.InputTag("GlobalHaloData")
                             )

