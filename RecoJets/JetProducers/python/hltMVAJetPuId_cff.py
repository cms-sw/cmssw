from RecoJets.JetProducers.hltPUIdAlgo_cff import *
import RecoJets.JetProducers.MVAJetPuIdProducer_cfi as _mod

hltMVAJetPuIdCalculator = _mod.MVAJetPuIdProducer.clone(
    produceJetIds = False,
    algos   = cms.VPSet(cms.VPSet(full_74x)),
    jec     = "AK4PFchs"
)
hltMVAJetPuIdEvaluator = hltMVAJetPuIdCalculator.clone(
    jetids = "pileupJetIdCalculator" 
)
