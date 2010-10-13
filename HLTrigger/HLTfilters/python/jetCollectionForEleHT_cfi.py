import FWCore.ParameterSet.Config as cms


hltMCJetCorJetIcone5HF07EleRemoved = cms.EDProducer('JetCollectionForEleHT',
                                            HltElectronTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt10HT70PixelMatchFilter"),
                                            SourceJetTag = cms.InputTag("hltMCJetCorJetIcone5HF07"),
                                            minDeltaR = cms.double(0.5)
)
