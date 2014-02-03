import FWCore.ParameterSet.Config as cms

CandIsolatorFromDeposits = cms.EDProducer("CandIsolatorFromDeposits",
     deposits = cms.VPSet(
             cms.PSet(
             src = cms.InputTag("hltMuPFIsoDepositCharged"),
             deltaR = cms.double(0.3),
             weight = cms.string('1'),
             vetos = cms.vstring('0.0001','Threshold(0.0)'),
             skipDefaultVeto = cms.bool(True),
             mode = cms.string('sum')
             )
      )
 )
