import FWCore.ParameterSet.Config as cms

ipCalib = cms.EDAnalyzer("ImpactParameterCalibration",
                                 
                                 jetTagsColl              = cms.string("jetProbabilityBJetTags"),
                                 writeToDB                = cms.bool(False),
                                 writeToBinary            = cms.bool(False),
                                 nBins                    = cms.int32(10000),
                                 maxSignificance          = cms.double(50.0),
                                 writeToRootXML           = cms.bool(True),
                                 tagInfoSrc               = cms.InputTag("impactParameterTagInfos"),
                                 inputCategories          = cms.string('HardCoded'),
                                 primaryVertexSrc         = cms.InputTag("offlinePrimaryVertices"),
                                 Jets                     = cms.InputTag('selectedPatJetsPF2PAT'),
                                 jetPModuleName           = cms.string('jetProbabilityBJetTags'),
                                 produceJetProbaTree      = cms.bool(True),
                                 jetCorrector             = cms.string('ak5PFL1FastL2L3'),
                                 MinPt                    = cms.double(10.0),
                                 MaxEta                   = cms.double(2.5)    
                                 
                                 )
