# The following comments couldn't be translated into the new config version:

# DQM services

import FWCore.ParameterSet.Config as cms

process = cms.Process("TBH42006Valid")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# geometry (Only Ecal)
process.load("Geometry.EcalTestBeam.TBH4GeometryXML_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

# Condition objects access
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetrieverTB_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# ECAL hits validation sequence
process.load("Validation.EcalHits.ecalSimHitsValidation_cfi")

process.load("Validation.EcalHits.ecalBarrelSimHitsValidation_cfi")

# ECAL digis validation sequence
process.load("Validation.EcalDigis.ecalDigisValidation_cfi")

process.load("Validation.EcalDigis.ecalBarrelDigisValidation_cfi")

# ECAL rechits validation sequence
process.load("Validation.EcalRecHits.ecalRecHitsValidation_cfi")

process.load("Validation.EcalRecHits.ecalBarrelRecHitsValidation_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:ECALH4TB_detsim_hits.root')
)

process.DaqMonitorROOTBackEnd = cms.Service("DaqMonitorROOTBackEnd")

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.ecalSimHitsValidationSequence = cms.Sequence(process.ecalSimHitsValidation*process.ecalBarrelSimHitsValidation)
process.ecalUnsuppressedDigisValidationSequence = cms.Sequence(process.ecalDigisValidation*process.ecalBarrelDigisValidation)
process.ecalUnsuppressedRecHitsValidationSequence = cms.Sequence(process.ecalRecHitsValidation*process.ecalBarrelRecHitsValidation)
process.p1 = cms.Path(process.ecalSimHitsValidationSequence*process.ecalUnsuppressedDigisValidationSequence*process.ecalUnsuppressedRecHitsValidationSequence)
process.CaloGeometryBuilder.SelectedCalos = ['EcalBarrel']
process.ecalDigisValidation.EBdigiCollection = 'simEcalUnsuppressedDigis'
process.ecalDigisValidation.EEdigiCollection = 'simEcalUnsuppressedDigis'
process.ecalDigisValidation.ESdigiCollection = 'simEcalUnsuppressedDigis'
process.ecalBarrelDigisValidation.EBdigiCollection = 'simEcalUnsuppressedDigis'
process.ecalRecHitsValidation.EBrechitCollection = cms.InputTag("ecalTBSimRecHit","EcalRecHitsEB")
process.ecalRecHitsValidation.EBuncalibrechitCollection = cms.InputTag("ecalTBSimWeightUncalibRecHit","EcalUncalibRecHitsEB")
process.ecalBarrelRecHitsValidation.EBdigiCollection = 'simEcalUnsuppressedDigis'
process.ecalBarrelRecHitsValidation.EBuncalibrechitCollection = cms.InputTag("ecalTBSimWeightUncalibRecHit","EcalUncalibRecHitsEB")


