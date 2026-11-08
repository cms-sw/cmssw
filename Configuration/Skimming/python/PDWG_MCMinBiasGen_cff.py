# Minbias generator skims
# 1) 1 muon with pt > 7, |eta|<2.5 --> 1Mu
# 2) 2 muons with pt > 2 and 2, |eta|<2.5 --> 2Mu
# 3) 2 muons with pt > 4 and 3, |eta|<2.5, opposite sign, mass [0.2,8.5] --> OS2Mu
# 4) 2 muons with pt > 4 and 4, |eta|<2.5, opposite sign, mass [4.9,5.9] --> OS2MuB
# 5) 3 muons with pt > 5/2/2, |eta|<2.5 --> 3Mu

import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import RAWSIMEventContent
MinBiasGenSkimFilter1 = cms.EDFilter("MCMultiParticleFilter",
          Status = cms.vint32(1),
          ParticleID = cms.vint32(13),
          PtMin = cms.vdouble(7.0),
          NumRequired = cms.int32(1),
          EtaMax = cms.vdouble(2.5),
          AcceptMore = cms.bool(True)
      )
MinBiasGenSkimFilter2 = cms.EDFilter("MCMultiParticleFilter",
          Status = cms.vint32(1),
          ParticleID = cms.vint32(13),
          PtMin = cms.vdouble(2.0),
          NumRequired = cms.int32(2),
          EtaMax = cms.vdouble(2.5),
          AcceptMore = cms.bool(True)
      )
MinBiasGenSkimFilter3 = cms.EDFilter("MCParticlePairFilter",
          Status = cms.untracked.vint32(1, 1),
          MinPt = cms.untracked.vdouble(4., 3.),
          MaxEta = cms.untracked.vdouble(2.5, 2.5),
          MinEta = cms.untracked.vdouble(-2.5, -2.5),
          ParticleCharge = cms.untracked.int32(-1),
          MaxInvMass = cms.untracked.double(8.5),
          MinInvMass = cms.untracked.double(0.2),
          ParticleID1 = cms.untracked.vint32(13),
          ParticleID2 = cms.untracked.vint32(13)
      )
MinBiasGenSkimFilter4 = cms.EDFilter("MCParticlePairFilter",
          Status = cms.untracked.vint32(1, 1),
          MinPt = cms.untracked.vdouble(4., 4.),
          MaxEta = cms.untracked.vdouble(2.5, 2.5),
          MinEta = cms.untracked.vdouble(-2.5, -2.5),
          ParticleCharge = cms.untracked.int32(-1),
          MaxInvMass = cms.untracked.double(5.9),
          MinInvMass = cms.untracked.double(4.9),
          ParticleID1 = cms.untracked.vint32(13),
          ParticleID2 = cms.untracked.vint32(13)
      )
MinBiasGenSkimFilter5 = cms.EDFilter("MCMultiParticleFilter",
          Status = cms.vint32(1),
          ParticleID = cms.vint32(13),
          PtMin = cms.vdouble(2.0),
          NumRequired = cms.int32(3),
          EtaMax = cms.vdouble(2.5),
          AcceptMore = cms.bool(True)
      )
MinBiasGenSkimFilter6 = cms.EDFilter("MCMultiParticleFilter",
          Status = cms.vint32(1),
          ParticleID = cms.vint32(13),
          PtMin = cms.vdouble(5.0),
          NumRequired = cms.int32(1),
          EtaMax = cms.vdouble(2.5),
          AcceptMore = cms.bool(True)
      )

MinBiasGenSkimSeq1Mu = cms.Sequence( MinBiasGenSkimFilter1 )
MinBiasGenSkimSeq2Mu = cms.Sequence( MinBiasGenSkimFilter2 )
MinBiasGenSkimSeqOS2Mu = cms.Sequence( MinBiasGenSkimFilter3 )
MinBiasGenSkimSeqOS2MuB = cms.Sequence( MinBiasGenSkimFilter4 )
MinBiasGenSkimSeq3Mu = cms.Sequence( MinBiasGenSkimFilter5 * MinBiasGenSkimFilter6)
