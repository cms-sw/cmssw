import FWCore.ParameterSet.Config as cms

#--------------------------
# DQM Module
#--------------------------
# just put that for the time being

castorOfflineMonitor = cms.EDAnalyzer("CastorMonitorModule",
                           ### GLOBAL VARIABLES
                           debug = cms.untracked.int32(0), # make debug an int so that different
                                                           # values can trigger different levels of messaging
                           # Turn on/off timing diagnostic info
                           showTiming          = cms.untracked.bool(False),
                           dump2database       = cms.untracked.bool(False),
                           pedestalsInFC = cms.untracked.bool(False),
			   
			   # Define Labels
                           digiLabel            = cms.InputTag("castorDigis"),
                           rawLabel             = cms.InputTag("rawDataCollector"),
                           unpackerReportLabel  = cms.InputTag("castorDigis"),
                           CastorRecHitLabel    = cms.InputTag("castorreco"),
                           CastorTowerLabel     = cms.InputTag("CastorTowerReco"),
                           CastorBasicJetsLabel = cms.InputTag("ak7CastorJets"),
                           CastorJetIDLabel     = cms.InputTag("ak7CastorJetID"),
			  
			 
                           DataIntMonitor= cms.untracked.bool(True),
                           TowerJetMonitor= cms.untracked.bool(True),

                           DigiMonitor = cms.untracked.bool(True),
                           DigiPerChannel = cms.untracked.bool(True), 
                           DigiInFC = cms.untracked.bool(False),
                          
                           RecHitMonitor = cms.untracked.bool(True), 
			   RecHitsPerChannel = cms.untracked.bool(True),

                           ChannelQualityMonitor= cms.untracked.bool(True),
                           nThreshold = cms.untracked.double(60),
                           dThreshold = cms.untracked.double(1.0),
                           OfflineMode = cms.untracked.bool(False),
                           averageEnergyMethod = cms.untracked.bool(True),          

                           PSMonitor= cms.untracked.bool(True),
                           numberSigma = cms.untracked.double(1.5),
                           thirdRegionThreshold =  cms.untracked.double(100),            
                           EDMonitor= cms.untracked.bool(False),
                           HIMonitor= cms.untracked.bool(True),
                                      
                           diagnosticPrescaleTime = cms.untracked.int32(-1),
                           diagnosticPrescaleUpdate = cms.untracked.int32(-1),
                           diagnosticPrescaleLS = cms.untracked.int32(-1),
                             
                           LEDMonitor = cms.untracked.bool(True),
                           LEDPerChannel = cms.untracked.bool(True),
                           FirstSignalBin = cms.untracked.int32(0),
                           LastSignalBin = cms.untracked.int32(9),
                           LED_ADC_Thresh = cms.untracked.double(-1000.0)      
                           )



