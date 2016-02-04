import FWCore.ParameterSet.Config as cms

# from CommonTools.ParticleFlow.Isolation.isoValMuonWithCharged_cfi import *
# from CommonTools.ParticleFlow.Isolation.isoValMuonWithNeutral_cfi import *
# from CommonTools.ParticleFlow.Isolation.isoValMuonWithPhotons_cfi import *

# compute isolation values, from various collections of iso deposits.
# could add the electrons and the muons as isolators, just to check that everything is fine

# pfMuonIsolationFromDepositsSequence = cms.Sequence(
#    isoValMuonWithCharged  +
#    isoValMuonWithNeutral  +
#    isoValMuonWithPhotons
#)


muPFIsoValueCharged03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositCharged"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
            )
     )
)

muPFIsoValueChargedAll03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositChargedAll"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
     )
   )
)

muPFIsoValueGamma03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)

muPFIsoValueNeutral03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
    )
 )
)

muPFIsoValuePU03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositPU"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)



muPFIsoValueCharged04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositCharged"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
            )
     )
)




muPFIsoValueChargedAll04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositChargedAll"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
     )
   )
)

muPFIsoValueGamma04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositGamma"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)


muPFIsoValueNeutral04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositNeutral"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
    )
 )

)
muPFIsoValuePU04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositPU"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)


pfMuonIsolationFromDepositsSequence =  cms.Sequence(
    # muPFIsoDepositCharged+
    # muPFIsoDepositChargedAll+
    # muPFIsoDepositGamma+
    # muPFIsoDepositNeutral+
    # muPFIsoDepositPU+
    ##############################
    muPFIsoValueCharged03+
    muPFIsoValueChargedAll03+
    muPFIsoValueGamma03+
    muPFIsoValueNeutral03+
    muPFIsoValuePU03+
    ############################## 
    muPFIsoValueCharged04+
    muPFIsoValueChargedAll04+
    muPFIsoValueGamma04+
    muPFIsoValueNeutral04+
    muPFIsoValuePU04
    )
