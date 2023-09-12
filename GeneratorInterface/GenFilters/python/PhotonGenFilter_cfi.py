import FWCore.ParameterSet.Config as cms

photonGenFilter = cms.EDFilter('PhotonGenFilter',
                      MaxEta = cms.untracked.double(2.4),    
                      MinEta = cms.untracked.double(-2.4),
                      MinPt = cms.untracked.double(20.),    
                      drMin = cms.untracked.double(0.1),
                      ptThreshold = cms.untracked.double(2.)
                  )