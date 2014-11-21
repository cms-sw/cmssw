import FWCore.ParameterSet.Config as cms



phPFIsoValueCharged03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositChargedPAT"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
            )
     )
)

phPFIsoValueChargedAll03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositChargedAllPAT"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
     )
   )
)

phPFIsoValueGamma03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositGammaPAT"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('EcalEndcaps:ConeVeto(0.05)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
      )
   )
)

phPFIsoValueNeutral03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositNeutralPAT"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
        )
    )
)

phPFIsoValuePU03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositPUPAT"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
      )
   )
)



phPFIsoValueCharged04PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositChargedPAT"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
            )
     )
)




phPFIsoValueChargedAll04PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositChargedAllPAT"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
     )
   )
)

phPFIsoValueGamma04PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositGammaPAT"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('EcalEndcaps:ConeVeto(0.05)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
      )
   )
)


phPFIsoValueNeutral04PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositNeutralPAT"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
    )
 )
)

phPFIsoValuePU04PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositPUPAT"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
      )
   )
)

photonPFIsolationValuesPATSequence = (
    phPFIsoValueCharged03PFIdPAT+
    phPFIsoValueChargedAll03PFIdPAT+
    phPFIsoValueGamma03PFIdPAT+
    phPFIsoValueNeutral03PFIdPAT+
    phPFIsoValuePU03PFIdPAT+
    ##############################
    phPFIsoValueCharged04PFIdPAT+
    phPFIsoValueChargedAll04PFIdPAT+
    phPFIsoValueGamma04PFIdPAT+
    phPFIsoValueNeutral04PFIdPAT+
    phPFIsoValuePU04PFIdPAT
    )
