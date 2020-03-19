import FWCore.ParameterSet.Config as cms



phPFIsoValueCharged03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositCharged"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
            )
     )
)

phPFIsoValueChargedAll03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositChargedAll"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
     )
   )
)

phPFIsoValueGamma03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('EcalEndcaps:ConeVeto(0.05)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
      )
   )
)

phPFIsoValueNeutral03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
        )
    )
)

phPFIsoValuePU03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositPU"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
      )
   )
)



phPFIsoValueCharged04PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositCharged"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
            )
     )
)




phPFIsoValueChargedAll04PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositChargedAll"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
     )
   )
)

phPFIsoValueGamma04PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositGamma"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('EcalEndcaps:ConeVeto(0.05)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
      )
   )
)


phPFIsoValueNeutral04PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositNeutral"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
    )
 )
)

phPFIsoValuePU04PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("phPFIsoDepositPU"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
      )
   )
)

photonPFIsolationValuesTask = cms.Task(
    phPFIsoValueCharged03PFId,
    phPFIsoValueChargedAll03PFId,
    phPFIsoValueGamma03PFId,
    phPFIsoValueNeutral03PFId,
    phPFIsoValuePU03PFId,
    ############################## 
    phPFIsoValueCharged04PFId,
    phPFIsoValueChargedAll04PFId,
    phPFIsoValueGamma04PFId,
    phPFIsoValueNeutral04PFId,
    phPFIsoValuePU04PFId
    )
photonPFIsolationValuesSequence = cms.Sequence(photonPFIsolationValuesTask)
