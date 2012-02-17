import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.electronIdMVABased_cfi import *

electronsWithPresel = cms.EDFilter("GsfElectronSelector",
                                   src = cms.InputTag("ecalDrivenGsfElectrons"),
                                   cut = cms.string("pt > 10 && ecalDrivenSeed && passingCutBasedPreselection"),
                                   )

mvaElectrons = cms.EDFilter("ElectronIdMVABased",
                            vertexTag = cms.InputTag('offlinePrimaryVertices'),
                            electronTag = cms.InputTag('electronsWithPresel'),
                            HZZmvaWeightFile = cms.string('RecoEgamma/ElectronIdentification/data/TMVA_BDTSimpleCat_17Feb2011.weights.xml'),
                            thresholdBarrel = cms.double( -0.1875 ),
                            thresholdEndcap = cms.double( -0.1075 ),
                            )

