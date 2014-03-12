from RecoJets.JetProducers.puJetIDAlgo_cff import *

pileupJetIdCalculator = cms.EDProducer('PileupJetIdProducer',
                                       produceJetIds = cms.bool(True),
                                       jetids = cms.InputTag(""),
                                       runMvas = cms.bool(False),
                                       jets = cms.InputTag("ak5PFJetsCHS"),
                                       vertexes = cms.InputTag("offlinePrimaryVertices"),
                                       algos = cms.VPSet(cms.VPSet(cutbased)),
                                       
                                       rho     = cms.InputTag("kt6PFJets","rho"),
                                       jec     = cms.string("AK5PFchs"),
                                       applyJec = cms.bool(False),
                                       inputIsCorrected = cms.bool(True),
                                       residualsFromTxt = cms.bool(False),
                                       residualsTxt     = cms.FileInPath("RecoJets/JetProducers/data/dummy.txt"),
                                       )

pileupJetIdEvaluator = cms.EDProducer('PileupJetIdProducer',
                                      produceJetIds = cms.bool(False),
                                      jetids = cms.InputTag("pileupJetIdCalculator"),
                                      runMvas = cms.bool(True),
                                      jets = cms.InputTag("ak5PFJetsCHS"),
                                      vertexes = cms.InputTag("offlinePrimaryVertices"),
                                      algos = cms.VPSet(cms.VPSet(cutbased,full_53x_chs)),
                                      
                                      rho     = cms.InputTag("kt6PFJets","rho"),
                                      jec     = cms.string("AK5PFchs"),
                                      applyJec = cms.bool(False),
                                      inputIsCorrected = cms.bool(True),
                                      residualsFromTxt = cms.bool(False),
                                      residualsTxt     = cms.FileInPath("RecoJets/JetProducers/data/dummy.txt"),
                                      )
