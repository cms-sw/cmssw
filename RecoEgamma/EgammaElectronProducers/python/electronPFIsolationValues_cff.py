import FWCore.ParameterSet.Config as cms

elPFIsoValueCharged03 = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("elPFIsoDepositCharged"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
            )
     )
)

elPFIsoValueChargedAll03 = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("elPFIsoDepositChargedAll"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
     )
   )
)

elPFIsoValueGamma03 = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("elPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('EcalEndcaps:ConeVeto(0.08)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
      )
   )
)

elPFIsoValueNeutral03 = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("elPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
            )
        )
    )

elPFIsoValuePU03 = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("elPFIsoDepositPU"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
            )
   )
)

elPFIsoValueCharged04 = elPFIsoValueCharged03.clone()
elPFIsoValueCharged04.deposits[0].deltaR = cms.double(0.4)


elPFIsoValueChargedAll04 = elPFIsoValueChargedAll03.clone()
elPFIsoValueChargedAll04.deposits[0].deltaR = cms.double(0.4)

elPFIsoValueGamma04 = elPFIsoValueGamma03.clone()
elPFIsoValueGamma04.deposits[0].deltaR = cms.double(0.4)


elPFIsoValueNeutral04 = elPFIsoValueNeutral03.clone()
elPFIsoValueNeutral04.deposits[0].deltaR = cms.double(0.4)

elPFIsoValuePU04 = elPFIsoValuePU03.clone()
elPFIsoValuePU04.deposits[0].deltaR = cms.double(0.4)



electronPFIsolationValuesSequence = cms.Sequence (
    elPFIsoValueCharged03+
    elPFIsoValueChargedAll03+
    elPFIsoValueGamma03+
    elPFIsoValueNeutral03+
    elPFIsoValuePU03+
    ##############################
    elPFIsoValueCharged04+
    elPFIsoValueChargedAll04+
    elPFIsoValueGamma04+
    elPFIsoValueNeutral04+
    elPFIsoValuePU04
)
