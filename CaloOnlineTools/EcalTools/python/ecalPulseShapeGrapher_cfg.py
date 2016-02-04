import FWCore.ParameterSet.Config as cms

process = cms.Process("ECALPULSESHAPEGRAPHER")

process.load("FWCore.MessageService.MessageLogger_cfi")


process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
#'/store/data/Commissioning08/BeamHalo/RAW/GRtoBeam_v1/000/062/063/46D299E0-137F-DD11-9A92-001617E30CE8.root'
#'/store/data/Commissioning08/BeamHalo/RAW/GRtoBeam_v1/000/062/095/5A984886-717F-DD11-A16F-000423D6CA6E.root'
      '/store/data/Commissioning08/BeamHalo/RAW/GRtoBeam_v1/000/062/096/56077B20-7B7F-DD11-B9D5-00161757BF42.root'
)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(15) )

process.load("CaloOnlineTools.EcalTools.ecalPulseShapeGrapher_cfi")

process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.EcalCommonData.EcalOnly_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")
    
import RecoLocalCalo.EcalRecProducers.ecalMaxSampleUncalibRecHit_cfi
#process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalMaxSampleUncalibRecHit_cfi.ecalMaxSampleUncalibRecHit.clone()
import RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi
process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi.ecalFixedAlphaBetaFitUncalibRecHit.clone()
process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.MessageLogger = cms.Service("MessageLogger",
    #suppressInfo = cms.untracked.vstring('ecalEBunpacker'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    categories = cms.untracked.vstring('EcalPulseShapeGrapher'),
    destinations = cms.untracked.vstring('cout')
)

#process.TFileService = cms.Service("TFileService",
#  fileName = cms.string('ecalBxOrbitNumberGrapher.root'),
#  closeFileFast = cms.untracked.bool(True)
#)

process.p = cms.Path(process.ecalEBunpacker*process.ecalUncalibHit*process.ecalRecHit*process.ecalPulseShapeGrapher)

process.GlobalTag.globaltag = 'CRUZET4_V5P::All'
process.ecalUncalibHit.EBdigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalUncalibHit.EEdigiCollection = 'ecalEBunpacker:eeDigis'
process.ecalRecHit.ChannelStatusToBeExcluded = [1]
process.ecalRecHit.EBuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEB'
process.ecalRecHit.EEuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEE'
process.EcalTrivialConditionRetriever.producedEcalWeights = False
process.EcalTrivialConditionRetriever.producedEcalPedestals = False
process.EcalTrivialConditionRetriever.producedEcalIntercalibConstants = False
process.EcalTrivialConditionRetriever.producedEcalIntercalibErrors = False
process.EcalTrivialConditionRetriever.producedEcalGainRatios = False
process.EcalTrivialConditionRetriever.producedEcalADCToGeVConstant = False
process.EcalTrivialConditionRetriever.producedEcalLaserCorrection = False
process.EcalTrivialConditionRetriever.producedChannelStatus = cms.untracked.bool(False)
#process.EcalTrivialConditionRetriever.producedChannelStatus = True
#process.EcalTrivialConditionRetriever.channelStatusFile = 'CaloOnlineTools/EcalTools/data/listCRUZET4.v5.hashed.txt'
#es_prefer_EcalChannelStatus = cms.ESPrefer("EcalTrivialConditionRetriever","EcalChannelStatus")
