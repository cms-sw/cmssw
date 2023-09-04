import FWCore.ParameterSet.Config as cms

photonGenFilter = cms.EDFilter('PhotonGenFilter',
                      MaxEta = cms.untracked.vdouble(2.4),    
                      MinEta = cms.untracked.vdouble(-2.4),
                      MinPt = cms.untracked.vdouble(20),    
                      Status = cms.untracked.vint32(1),    
                      ParticleID = cms.untracked.vint32(22),
                      drMin = cms.untracked.vdouble(0.1)
                  )