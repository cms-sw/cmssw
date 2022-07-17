import FWCore.ParameterSet.Config as cms

# The following should be removed up to  <--- when moving to GED only
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

elPFIsoValueCharged04 = elPFIsoValueCharged03.clone(deposits = {0: dict(deltaR = 0.4)} )
elPFIsoValueChargedAll04 = elPFIsoValueChargedAll03.clone(deposits = {0: dict(deltaR = 0.4)} )
elPFIsoValueGamma04 = elPFIsoValueGamma03.clone(deposits = {0: dict(deltaR = 0.4)} )
elPFIsoValueNeutral04 = elPFIsoValueNeutral03.clone(deposits = {0: dict(deltaR = 0.4)} )
elPFIsoValuePU04 = elPFIsoValuePU03.clone(deposits ={0: dict(deltaR = 0.4)} )

electronPFIsolationValuesTask = cms.Task(
    elPFIsoValueCharged03,
    elPFIsoValueChargedAll03,
    elPFIsoValueGamma03,
    elPFIsoValueNeutral03,
    elPFIsoValuePU03,
    ##############################
    elPFIsoValueCharged04,
    elPFIsoValueChargedAll04,
    elPFIsoValueGamma04,
    elPFIsoValueNeutral04,
    elPFIsoValuePU04
)
electronPFIsolationValuesSequence = cms.Sequence(electronPFIsolationValuesTask)

#<----------------
