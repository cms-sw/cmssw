import FWCore.ParameterSet.Config as cms

mvaElectrons = cms.EDFilter("ElectronIdMVABased",
                            vertexTag = cms.InputTag('offlinePrimaryVertices'),
                            electronTag = cms.InputTag('gsfElectrons'),
                            HZZmvaWeightFile = cms.string('RecoEgamma/ElectronIdentification/data/TMVA_BDTSimpleCat_17Feb2011.weights.xml'),
                            thresholdBarrel = cms.double( -0.1875 ),
                            thresholdEndcap = cms.double( -0.1075 ),
                            )
