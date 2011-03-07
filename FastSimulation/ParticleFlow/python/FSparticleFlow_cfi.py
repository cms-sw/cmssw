import FWCore.ParameterSet.Config as cms

FSparticleFlow = cms.EDProducer("FSPFProducer",

    # PFCandidate label
   pfCandidates = cms.InputTag("particleFlow"),
                                
   barrel_correction = cms.double(0.075),
   endcap_correction = cms.double(0.005),
   debug = cms.bool(False)
     
)



