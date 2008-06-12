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

gamIsoFromDepsEcalSCVetoFromClusts = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        mode = cms.string('sum'),
        src = cms.InputTag("gamIsoDepositEcalSCVetoFromClusts"),
        weight = cms.string('1'),
        deltaR = cms.double(0.4),
        vetos = cms.vstring(),
        skipDefaultVeto = cms.bool(True)
    ))
)

gamIsoFromDepsEcalFromClusts = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        mode = cms.string('sum'),
        src = cms.InputTag("gamIsoDepositEcalFromClusts"),
        weight = cms.string('1'),
        deltaR = cms.double(0.4),
        vetos = cms.vstring('EcalBarrel:0.045', 
                            'EcalBarrel:RectangularEtaPhiVeto(-0.01,0.01,-0.5,0.5)', 
                            'EcalEndcaps:0.070', 
                            'EcalEndcaps:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)'),
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
                            'EcalBarrel:Threshold(0.080)', 
                            'EcalEndcaps:0.070', 
                            'EcalEndcaps:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)', 
                            'EcalEndcaps:Threshold(0.30)'),
        skipDefaultVeto = cms.bool(True)
    ))
)

gamIsoFromDepsHcalFromHits = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        mode = cms.string('sum'),
        src = cms.InputTag("gamIsoDepositHcalFromHits"),
        weight = cms.string('1'),
        deltaR = cms.double(0.4),
        vetos = cms.vstring('0.0'),
        skipDefaultVeto = cms.bool(True)
    ))
)

gamIsoFromDepsHcalFromTowers = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("gamIsoDepositHcalFromTowers"),
        deltaR = cms.double(0.4),
        weight = cms.string('1'),
        vetos = cms.vstring('0.00'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)


