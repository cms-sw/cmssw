import FWCore.ParameterSet.Config as cms



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


muPFSumDRIsoValueCharged03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositCharged"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
            )
     )
)

muPFMeanDRIsoValueCharged03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositCharged"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
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

muPFSumDRIsoValueChargedAll03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositChargedAll"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
     )
   )
)

muPFMeanDRIsoValueChargedAll03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositChargedAll"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
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
muPFSumDRIsoValueGamma03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
      )
   )
)

muPFMeanDRIsoValueGamma03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
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
muPFSumDRIsoValueNeutral03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
    )
 )
)

muPFMeanDRIsoValueNeutral03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
    )
 )
)



muPFIsoValueGammaHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)
muPFSumDRIsoValueGammaHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
      )
   )
)

muPFMeanDRIsoValueGammaHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
      )
   )
)


muPFIsoValueNeutralHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
    )
 )
)

muPFMeanDRIsoValueNeutralHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
    )
 )
)

muPFSumDRIsoValueNeutralHighThreshold03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
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

muPFSumDRIsoValuePU03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositPU"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
      )
   )
)

muPFMeanDRIsoValuePU03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositPU"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
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

muPFSumDRIsoValueCharged04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositCharged"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
            )
     )
)

muPFMeanDRIsoValueCharged04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositCharged"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
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

muPFSumDRIsoValueChargedAll04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositChargedAll"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
     )
   )
)

muPFMeanDRIsoValueChargedAll04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositChargedAll"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
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

muPFSumDRIsoValueGamma04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositGamma"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
      )
   )
)

muPFMeanDRIsoValueGamma04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositGamma"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
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

muPFSumDRIsoValueNeutral04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositNeutral"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
    )
 )

)

muPFMeanDRIsoValueNeutral04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositNeutral"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
    )
 )

)


muPFIsoValueGammaHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositGamma"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)

muPFMeanDRIsoValueGammaHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositGamma"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
      )
   )
)

muPFSumDRIsoValueGammaHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositGamma"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
      )
   )
)


muPFIsoValueNeutralHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositNeutral"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
    )
 )

)

muPFMeanDRIsoValueNeutralHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositNeutral"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
    )
 )

)

muPFSumDRIsoValueNeutralHighThreshold04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositNeutral"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(1.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
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

muPFMeanDRIsoValuePU04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositPU"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('meanDR')
      )
   )
)

muPFSumDRIsoValuePU04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositPU"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sumDR')
      )
   )
)

muonPFIsolationValuesSequence = (
    muPFIsoValueCharged03+
    muPFMeanDRIsoValueCharged03+
    muPFSumDRIsoValueCharged03+
    muPFIsoValueChargedAll03+
    muPFMeanDRIsoValueChargedAll03+
    muPFSumDRIsoValueChargedAll03+
    muPFIsoValueGamma03+
    muPFMeanDRIsoValueGamma03+
    muPFSumDRIsoValueGamma03+
    muPFIsoValueNeutral03+
    muPFMeanDRIsoValueNeutral03+
    muPFSumDRIsoValueNeutral03+
    muPFIsoValueGammaHighThreshold03+
    muPFMeanDRIsoValueGammaHighThreshold03+
    muPFSumDRIsoValueGammaHighThreshold03+
    muPFIsoValueNeutralHighThreshold03+
    muPFMeanDRIsoValueNeutralHighThreshold03+
    muPFSumDRIsoValueNeutralHighThreshold03+
    muPFIsoValuePU03+
    muPFMeanDRIsoValuePU03+
    muPFSumDRIsoValuePU03+
    ############################## 
    muPFIsoValueCharged04+
    muPFMeanDRIsoValueCharged04+
    muPFSumDRIsoValueCharged04+
    muPFIsoValueChargedAll04+
    muPFMeanDRIsoValueChargedAll04+
    muPFSumDRIsoValueChargedAll04+
    muPFIsoValueGamma04+
    muPFMeanDRIsoValueGamma04+
    muPFSumDRIsoValueGamma04+
    muPFIsoValueNeutral04+
    muPFMeanDRIsoValueNeutral04+
    muPFSumDRIsoValueNeutral04+
    muPFIsoValueGammaHighThreshold04+
    muPFMeanDRIsoValueGammaHighThreshold04+
    muPFSumDRIsoValueGammaHighThreshold04+
    muPFIsoValueNeutralHighThreshold04+
    muPFMeanDRIsoValueNeutralHighThreshold04+
    muPFSumDRIsoValueNeutralHighThreshold04+
    muPFIsoValuePU04+
    muPFMeanDRIsoValuePU04+
    muPFSumDRIsoValuePU04
    )
