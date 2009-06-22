import FWCore.ParameterSet.Config as cms


from PhysicsTools.PFCandProducer.TopProjectors.noJet_cfi 


dump = cms.EDAnalyzer("EventContentAnalyzer")

pfNoJetSequence = cms.Sequence(
    jetsOnNoMuonsNoPileUp
    )

