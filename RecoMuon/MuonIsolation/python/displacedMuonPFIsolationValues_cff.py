import FWCore.ParameterSet.Config as cms



dispMuPFIsoValueCharged03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositCharged"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
            )
     )
)


dispMuPFSumDRIsoValueCharged03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositCharged"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
            )
     )
)

dispMuPFMeanDRIsoValueCharged03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositCharged"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
            )
     )
)


dispMuPFIsoValueChargedAll03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositChargedAll"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
     )
   )
)

dispMuPFSumDRIsoValueChargedAll03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositChargedAll"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
     )
   )
)

dispMuPFMeanDRIsoValueChargedAll03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositChargedAll"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
     )
   )
)


dispMuPFIsoValueGamma03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)
dispMuPFSumDRIsoValueGamma03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
      )
   )
)

dispMuPFMeanDRIsoValueGamma03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
      )
   )
)


dispMuPFIsoValueNeutral03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
    )
 )
)
dispMuPFSumDRIsoValueNeutral03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
    )
 )
)

dispMuPFMeanDRIsoValueNeutral03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
    )
 )
)



dispMuPFIsoValueGammaHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)
dispMuPFSumDRIsoValueGammaHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
      )
   )
)

dispMuPFMeanDRIsoValueGammaHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
      )
   )
)


dispMuPFIsoValueNeutralHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
    )
 )
)

dispMuPFMeanDRIsoValueNeutralHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
    )
 )
)

dispMuPFSumDRIsoValueNeutralHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
    )
 )
)

dispMuPFIsoValuePU03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositPU"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)

dispMuPFSumDRIsoValuePU03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositPU"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
      )
   )
)

dispMuPFMeanDRIsoValuePU03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositPU"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
      )
   )
)



dispMuPFIsoValueCharged04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositCharged"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
            )
     )
)

dispMuPFSumDRIsoValueCharged04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositCharged"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
            )
     )
)

dispMuPFMeanDRIsoValueCharged04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositCharged"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
            )
     )
)




dispMuPFIsoValueChargedAll04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositChargedAll"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
     )
   )
)

dispMuPFSumDRIsoValueChargedAll04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositChargedAll"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
     )
   )
)

dispMuPFMeanDRIsoValueChargedAll04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositChargedAll"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
     )
   )
)





dispMuPFIsoValueGamma04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositGamma"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)

dispMuPFSumDRIsoValueGamma04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositGamma"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
      )
   )
)

dispMuPFMeanDRIsoValueGamma04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositGamma"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
      )
   )
)


dispMuPFIsoValueNeutral04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositNeutral"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
    )
 )

)

dispMuPFSumDRIsoValueNeutral04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositNeutral"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
    )
 )

)

dispMuPFMeanDRIsoValueNeutral04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositNeutral"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
    )
 )

)


dispMuPFIsoValueGammaHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositGamma"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)

dispMuPFMeanDRIsoValueGammaHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositGamma"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
      )
   )
)

dispMuPFSumDRIsoValueGammaHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositGamma"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
      )
   )
)


dispMuPFIsoValueNeutralHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositNeutral"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
    )
 )

)

dispMuPFMeanDRIsoValueNeutralHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositNeutral"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
    )
 )

)

dispMuPFSumDRIsoValueNeutralHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositNeutral"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
    )
 )

)

dispMuPFIsoValuePU04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositPU"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)

dispMuPFMeanDRIsoValuePU04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositPU"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
      )
   )
)

dispMuPFSumDRIsoValuePU04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("dispMuPFIsoDepositPU"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
      )
   )
)

displacedMuonPFIsolationValuesTask = cms.Task(
    dispMuPFIsoValueCharged03,
    dispMuPFMeanDRIsoValueCharged03,
    dispMuPFSumDRIsoValueCharged03,
    dispMuPFIsoValueChargedAll03,
    dispMuPFMeanDRIsoValueChargedAll03,
    dispMuPFSumDRIsoValueChargedAll03,
    dispMuPFIsoValueGamma03,
    dispMuPFMeanDRIsoValueGamma03,
    dispMuPFSumDRIsoValueGamma03,
    dispMuPFIsoValueNeutral03,
    dispMuPFMeanDRIsoValueNeutral03,
    dispMuPFSumDRIsoValueNeutral03,
    dispMuPFIsoValueGammaHighThreshold03,
    dispMuPFMeanDRIsoValueGammaHighThreshold03,
    dispMuPFSumDRIsoValueGammaHighThreshold03,
    dispMuPFIsoValueNeutralHighThreshold03,
    dispMuPFMeanDRIsoValueNeutralHighThreshold03,
    dispMuPFSumDRIsoValueNeutralHighThreshold03,
    dispMuPFIsoValuePU03,
    dispMuPFMeanDRIsoValuePU03,
    dispMuPFSumDRIsoValuePU03,
    ############################## 
    dispMuPFIsoValueCharged04,
    dispMuPFMeanDRIsoValueCharged04,
    dispMuPFSumDRIsoValueCharged04,
    dispMuPFIsoValueChargedAll04,
    dispMuPFMeanDRIsoValueChargedAll04,
    dispMuPFSumDRIsoValueChargedAll04,
    dispMuPFIsoValueGamma04,
    dispMuPFMeanDRIsoValueGamma04,
    dispMuPFSumDRIsoValueGamma04,
    dispMuPFIsoValueNeutral04,
    dispMuPFMeanDRIsoValueNeutral04,
    dispMuPFSumDRIsoValueNeutral04,
    dispMuPFIsoValueGammaHighThreshold04,
    dispMuPFMeanDRIsoValueGammaHighThreshold04,
    dispMuPFSumDRIsoValueGammaHighThreshold04,
    dispMuPFIsoValueNeutralHighThreshold04,
    dispMuPFMeanDRIsoValueNeutralHighThreshold04,
    dispMuPFSumDRIsoValueNeutralHighThreshold04,
    dispMuPFIsoValuePU04,
    dispMuPFMeanDRIsoValuePU04,
    dispMuPFSumDRIsoValuePU04
    )
displacedMuonPFIsolationValuesSequence = cms.Sequence(displacedMuonPFIsolationValuesTask)
