import FWCore.ParameterSet.Config as cms

hcalDetDiagNoiseMonitor=cms.EDAnalyzer("HcalDetDiagNoiseMonitor",
                                       # base class stuff
                                       debug                  = cms.untracked.int32(0),
                                       online                 = cms.untracked.bool(False),
                                       AllowedCalibTypes      = cms.untracked.vint32(0,1,2,3,4,5),  # should noise monitor only look at non-calib events?
                                       mergeRuns              = cms.untracked.bool(False),
                                       enableCleanup          = cms.untracked.bool(False),
                                       subSystemFolder        = cms.untracked.string("Hcal/"),
                                       TaskFolder             = cms.untracked.string("DetDiagNoiseMonitor_Hcal/"),
                                       skipOutOfOrderLS       = cms.untracked.bool(True),
                                       NLumiBlocks            = cms.untracked.int32(4000),
                                       makeDiagnostics        = cms.untracked.bool(False),
                                       
                                       # DetDiag Noise Monitor-specific Info
                                       UseDB                  = cms.untracked.bool(False),
 
                                       RawDataLabel           = cms.untracked.InputTag("rawDataCollector"),
                                       digiLabel              = cms.untracked.InputTag("hcalDigis"),
                        	       gtLabel                = cms.untracked.InputTag("l1GtUnpack"),
                                       hcalTBTriggerDataTag   = cms.InputTag("tbunpack"),
                                       Overwrite              = cms.untracked.bool(True),
                                       # path to store datasets for current run
                                       OutputFilePath         = cms.untracked.string(""),
                                       # path to store xmz.zip file to be uploaded into OMDG
                                       XmlFilePath            = cms.untracked.string(""),
                                       HPDthresholdHi         = cms.untracked.double(49.0),
                                       HPDthresholdLo         = cms.untracked.double(10.0),
                                       SpikeThreshold         = cms.untracked.double(0.5),
                                       
				       HLTriggerResults                    = cms.untracked.InputTag("TriggerResults","","HLT"),
                                       MetSource                           = cms.untracked.InputTag("met"),
                                       JetSource                           = cms.untracked.InputTag("iterativeCone5CaloJets"),
                                       TrackSource                         = cms.untracked.InputTag("generalTracks"),
                                       VertexSource                        = cms.untracked.InputTag("offlinePrimaryVertices"),
                                       UseVertexCuts		           = cms.untracked.bool(True),
                                       rbxCollName                         = cms.untracked.string('hcalnoise'),
                                       MonitoringTriggerRequirement        = cms.untracked.string("HLT_MET100"),
                                       PhysDeclaredRequirement		   = cms.untracked.string("HLT_PhysicsDeclared"),
                                       UseMonitoringTrigger	           = cms.untracked.bool(False),
                                       JetMinEt                            = cms.untracked.double(10.0),
                                       JetMaxEta                           = cms.untracked.double(2.0),
                                       ConstituentsToJetMatchingDeltaR     = cms.untracked.double(0.5),
                                       TrackMaxIp                          = cms.untracked.double(0.1),
                                       TrackMinThreshold                   = cms.untracked.double(1.0),
                                       MinJetChargeFraction                = cms.untracked.double(0.05),
                                       MaxJetHadronicEnergyFraction        = cms.untracked.double(0.98),
                                       caloTowerCollName                   = cms.InputTag("towerMaker"),
                                   )
