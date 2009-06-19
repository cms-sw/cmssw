import FWCore.ParameterSet.Config as cms


from PhysicsTools.PFCandProducer.pfTopProjectionPFJetsOnPFCandidates_cfi import pfTopProjectionPFJetsOnPFCandidates  as tp
jetsOnNoMuonsNoPileUp = tp.clone()
jetsOnNoMuonsNoPileUp.name = 'jetsOnNoMuonsNoPileUp'
jetsOnNoMuonsNoPileUp.topCollection = 'pfJets'
jetsOnNoMuonsNoPileUp.bottomCollection = 'muonsOnNoPileUp'


dump = cms.EDAnalyzer("EventContentAnalyzer")

pfNoJetSequence = cms.Sequence(
    jetsOnNoMuonsNoPileUp
    )

