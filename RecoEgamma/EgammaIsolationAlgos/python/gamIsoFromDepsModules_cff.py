import FWCore.ParameterSet.Config as cms

gamIsoFromDepsTk = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        mode = cms.string('sum'),
        src = cms.InputTag("gamIsoDepositTk"),
        weight = cms.string('1'),
        deltaR = cms.double(0.3),
        vetos = cms.vstring('0.015','Threshold(1.0)'),
        skipDefaultVeto = cms.bool(True)
    ))
)

gamIsoFromDepsEcalFromHits = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        mode = cms.string('sum'),
        src = cms.InputTag("gamIsoDepositEcalFromHits"),
        weight = cms.string('1'),
        deltaR = cms.double(0.4),
        vetos = cms.vstring('EcalBarrel:0.045', 
                            'EcalBarrel:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)', 
                            'EcalBarrel:ThresholdFromTransverse(0.080)', 
                            'EcalEndcaps:0.070', 
                            'EcalEndcaps:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)', 
                            'EcalEndcaps:ThresholdFromTransverse(0.30)'),
        skipDefaultVeto = cms.bool(True)
    ))
)

gamIsoFromDepsHcalFromHits = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        mode = cms.string('sum'),
        src = cms.InputTag("gamIsoDepositHcalFromHits"),
        weight = cms.string('1'),
        deltaR = cms.double(0.4),
        vetos = cms.vstring('0.15'),
        skipDefaultVeto = cms.bool(True)
    ))
)

gamIsoFromDepsHcalFromTowers = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("gamIsoDepositHcalFromTowers"),
        deltaR = cms.double(0.4),
        weight = cms.string('1'),
        vetos = cms.vstring('0.15'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)

gamIsoFromDepsHcalDepth1FromTowers = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("gamIsoDepositHcalDepth1FromTowers"),
        deltaR = cms.double(0.4),
        weight = cms.string('1'),
        vetos = cms.vstring('0.15'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)

gamIsoFromDepsHcalDepth2FromTowers = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("gamIsoDepositHcalDepth2FromTowers"),
        deltaR = cms.double(0.4),
        weight = cms.string('1'),
        vetos = cms.vstring('0.15'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)


