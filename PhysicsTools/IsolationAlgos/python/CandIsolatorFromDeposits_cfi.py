import FWCore.ParameterSet.Config as cms

hltMuPFSumDRIsoValueCharged03 = cms.EDProducer("CandIsolatorFromDeposits",
     deposits = cms.VPSet(
             cms.PSet(
             src = cms.InputTag("ISODEPOSIT_COLLECTION"), # input collection of type reco::IsoDepositMap
             deltaR = cms.double(99999.), #cone size
             weight = cms.string('1'),
             vetos = cms.vstring('0.0001','Threshold(0.0)'),
             skipDefaultVeto = cms.bool(True),
             mode = cms.string('sum')
             )
      )
 )
