import FWCore.ParameterSet.Config as cms

process = cms.Process("COSMICSANALYSIS")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")
process.GlobalTag.globaltag = 'CRUZET4_V5P::All'

process.EcalTrivialConditionRetriever.producedEcalWeights = False
process.EcalTrivialConditionRetriever.producedEcalPedestals = False
process.EcalTrivialConditionRetriever.producedEcalIntercalibConstants = False
process.EcalTrivialConditionRetriever.producedEcalIntercalibErrors = False
process.EcalTrivialConditionRetriever.producedEcalGainRatios = False
process.EcalTrivialConditionRetriever.producedEcalADCToGeVConstant = False
#process.EcalTrivialConditionRetriever.producedEcalADCToGeVConstant = True
#process.EcalTrivialConditionRetriever.adcToGeVEBConstant = 0.035
#process.EcalTrivialConditionRetriever.adcToGeVEEConstant = 0.060
process.EcalTrivialConditionRetriever.producedEcalLaserCorrection = False
#process.EcalTrivialConditionRetriever.producedEcalChannelStatus = False
process.EcalTrivialConditionRetriever.producedEcalChannelStatus = cms.untracked.bool(False)
#process.EcalTrivialConditionRetriever.channelStatusFile = 'CaloOnlineTools/EcalTools/data/listCRUZET4.v5.hashed.txt'
#process.es_prefer_EcalTrivialConditionRetriever = cms.ESPrefer("EcalTrivialConditionRetriever")

process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.siStripPedestalFrontier = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
process.siStripPedestalFrontier.connect = 'frontier://PromptProd/CMS_COND_21X_STRIP'
process.siStripPedestalFrontier.toGet = cms.VPSet(cms.PSet(
            record = cms.string('SiStripPedestalsRcd'),
                        tag = cms.string('SiStripPedestals_TKCC_21X_v3_hlt')
                    ))
process.siStripPedestalFrontier.BlobStreamerName = 'TBufferBlobStreamingService'
process.es_prefer_SiStripFake = cms.ESPrefer("PoolDBESSource","siStripPedestalFrontier")

process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

import RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi
process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi.ecalFixedAlphaBetaFitUncalibRecHit.clone()
process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")

process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
process.load("RecoEcal.EgammaClusterProducers.cosmicClusteringSequence_cff")

process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtBoardMapsConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v3_Unprescaled_cff")
import FWCore.Modules.printContent_cfi
process.dumpEv = FWCore.Modules.printContent_cfi.printContent.clone()

import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
process.gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()
process.load("CaloOnlineTools.EcalTools.ecalCosmicsHists_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CommonTopologies.globalTrackingGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("RecoLocalCalo.Configuration.hcalLocalReco_cff")

process.load("HLTrigger.special.TriggerTypeFilter_cfi")

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = (cms.untracked.vstring(
        '/store/data/Commissioning08/BeamHalo/RAW/StuffAlmostToP5_v1/000/061/642/94D3CADF-A47D-DD11-BF3E-000423D94E1C.root'
))
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
    suppressWarning = cms.untracked.vstring('ecalEBunpacker','ecalUncalibHit','ecalRecHit',
                                            'hcalDigis','hcalLocalRecoSequence','ecalCosmicsHists'),
    suppressInfo = cms.untracked.vstring('ecalEBunpacker', 'ecalUncalibHit', 
                                         'ecalRecHit', 'hcalDigis','hcalLocalRecoSequence'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    categories = cms.untracked.vstring('EcalCosmicsHists','CosmicClusterAlgo','CosmicClusterProducer'),
    destinations = cms.untracked.vstring('cout')
)

process.p = cms.Path(process.triggerTypeFilter*process.ecalEBunpacker*process.ecalUncalibHit*process.ecalRecHit*process.cosmicClusteringSequence*process.gtDigis*process.ecalCosmicsHists)

process.ecalUncalibHit.EBdigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalUncalibHit.EEdigiCollection = 'ecalEBunpacker:eeDigis'
process.ecalRecHit.ChannelStatusToBeExcluded = [1]
process.ecalRecHit.EBuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEB'
process.ecalRecHit.EEuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEE'
process.cosmicBasicClusters.barrelUnHitProducer = "ecalUncalibHit"
process.cosmicBasicClusters.endcapUnHitProducer = "ecalUncalibHit"

process.gtDigis.DaqGtInputTag = 'source'

process.ecalCosmicsHists.fileName = 'EcalCosmicsHists'
process.ecalCosmicsHists.runInFileName = True
#process.ecalCosmicsHists.MinTimingAmpEB = 0.1,   # for adcToGeV=0.009, gain 200
#process.ecalCosmicsHists.MinRecHitAmpEB = 0.027, # for adcToGeV=0.009, gain 200
process.ecalCosmicsHists.MinTimingAmpEB = 0.35   # for adcToGeV=0.035, gain 50
process.ecalCosmicsHists.MinRecHitAmpEB = 0.070  # for adcToGeV=0.035, gain 50
process.ecalCosmicsHists.MinTimingAmpEE = 0.9    # for adcToGeV=0.06
process.ecalCosmicsHists.MinRecHitAmpEE = 0.180  # for adcToGeV=0.06

process.hbhereco = process.hbheprereco.clone()
process.hbhereco.firstSample = 1
process.hbhereco.samplesToAdd = 8
process.hbhereco.correctForTimeslew = True
process.hbhereco.correctForPhaseContainment = True
process.hbhereco.correctionPhaseNS = 10.0
process.horeco.firstSample = 1
process.horeco.samplesToAdd = 8
process.horeco.correctForTimeslew = True
process.horeco.correctForPhaseContainment = True
process.horeco.correctionPhaseNS = 10.
process.hfreco.firstSample = 1
process.hfreco.samplesToAdd = 8
process.hfreco.correctForTimeslew = True
process.hfreco.correctForPhaseContainment = True
process.hfreco.correctionPhaseNS = 10.
 
process.triggerTypeFilter.SelectedTriggerType = 1

