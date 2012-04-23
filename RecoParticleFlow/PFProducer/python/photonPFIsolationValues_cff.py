import FWCore.ParameterSet.Config as cms



phPFIsoValueCharged03PFId = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositCharged"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
            )
     )
)

phPFIsoValueChargedAll03PFId = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositChargedAll"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
     )
   )
)

phPFIsoValueGamma03PFId = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('ConeVeto(0.05)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)

phPFIsoValueNeutral03PFId = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
    )
 )
)

phPFIsoValuePU03PFId = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositPU"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)



phPFIsoValueCharged04PFId = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositCharged"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
            )
     )
)




phPFIsoValueChargedAll04PFId = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositChargedAll"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
     )
   )
)

phPFIsoValueGamma04PFId = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositGamma"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('ConeVeto(0.05)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)


phPFIsoValueNeutral04PFId = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositNeutral"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
    )
 )

)
phPFIsoValuePU04PFId = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositPU"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)

photonPFIsolationValuesSequence = (
    phPFIsoValueCharged03PFId+
    phPFIsoValueChargedAll03PFId+
    phPFIsoValueGamma03PFId+
    phPFIsoValueNeutral03PFId+
    phPFIsoValuePU03PFId+
    ############################## 
    phPFIsoValueCharged04PFId+
    phPFIsoValueChargedAll04PFId+
    phPFIsoValueGamma04PFId+
    phPFIsoValueNeutral04PFId+
    phPFIsoValuePU04PFId
    )
