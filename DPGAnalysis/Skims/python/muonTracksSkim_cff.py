import FWCore.ParameterSet.Config as cms

muonSkim=cms.EDFilter("CandViewCountFilter", 
                 src =cms.InputTag("muons"), minNumber = cms.uint32(1))
muonTracksSkim = cms.Sequence(muonSkim)
