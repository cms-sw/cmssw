import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.ParticleSelectors.ptMinPFJetSelector_cfi import ptMinPFJets as pfJets
from CommonTools.ParticleFlow.Tools.jetTools import jetAlgo


#allPfJets = RecoJets.JetProducers.ic5PFJets_cfi.iterativeCone5PFJets.clone()
pfJets = jetAlgo('AK4')

pfJetsPtrs = cms.EDProducer("PFJetFwdPtrProducer",
                            src=cms.InputTag("pfJets")
                            )

#pfJets.src = 'allPfJets'
#pfJets.ptMin = 10

pfJetTask = cms.Task(
#    allPfJets ,
    pfJets ,
    pfJetsPtrs 
    )
pfJetSequence = cms.Sequence(pfJetTask)
