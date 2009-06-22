import FWCore.ParameterSet.Config as cms


from PhysicsTools.PFCandProducer.TopProjectors.noMuon_cfi import * 
from PhysicsTools.PFCandProducer.TopProjectors.noElectron_cfi import * 


dump = cms.EDAnalyzer("EventContentAnalyzer")

pfNoLeptonSequence = cms.Sequence(
    noMuon
#    + noElectron
    )

