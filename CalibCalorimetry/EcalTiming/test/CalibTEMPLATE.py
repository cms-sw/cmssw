import FWCore.ParameterSet.Config as cms

process = cms.Process("ECALCALIBRATIONANALYZER")
process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerMapping_cfi")

process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")

process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

import RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi
process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi.ecalFixedAlphaBetaFitUncalibRecHit.clone()
process.load("CaloOnlineTools.EcalCalibrationAnalyzer.ecalCalibrationAnalyzer_cfi")

process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring('/store/data/Commissioning08/TestEnables/RAW/v1/000/064/724/2633C1DA-7894-DD11-9154-000423D94908.root',
                                                                          '/store/data/Commissioning08/TestEnables/RAW/v1/000/064/724/C4B97E1A-7A94-DD11-AF6C-001617C3B6CC.root',
                                                                          '/store/data/Commissioning08/TestEnables/RAW/v1/000/064/724/E08E7DE6-7794-DD11-9E5D-000423D98EC4.root',
                                                                          '/store/data/Commissioning08/TestEnables/RAW/v1/000/064/724/EE26B110-7E94-DD11-AA2C-000423D98930.root')
                            )

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
        )
process.MessageLogger = cms.Service("MessageLogger")

process.preScaler = cms.EDFilter("Prescaler",
                                     prescaleOffset = cms.int32(0),
                                     prescaleFactor = cms.int32(1)
                                 )

process.TFileService = cms.Service("TFileService",
                                       fileName = cms.string('Calib_RUNNUMBER.root')
                                   )

process.p = cms.Path(process.preScaler*process.ecalEBunpacker*process.ecalUncalibHit*process.ecalCalibrationAnalyzer)
process.ecalUncalibHit.EBdigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalUncalibHit.EEdigiCollection = 'ecalEBunpacker:eeDigis'

