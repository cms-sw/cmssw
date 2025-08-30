import FWCore.ParameterSet.Config as cms

l1tJUMPProducer = cms.EDProducer("L1JUMPProducer", 
                                RawMET = cms.InputTag("l1tMETPFProducer"),
                                L1PFJets = cms.InputTag("l1tSC4PFL1PuppiCorrectedEmulator"),
                                MinJetpT = cms.double(30),
                                MaxJetEta = cms.double(3.0),
) 