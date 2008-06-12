import FWCore.ParameterSet.Config as cms

eleIsoFromDepsTk = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        mode = cms.string('sum'),
        src = cms.InputTag("eleIsoDepositTk"),
        weight = cms.string('1'),
        deltaR = cms.double(0.3),
        vetos = cms.vstring('0.015','Threshold(1.0)'),
        skipDefaultVeto = cms.bool(True)
    ))
)

eleIsoFromDepsEcalSCVetoFromClusts = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        mode = cms.string('sum'),
        src = cms.InputTag("eleIsoDepositEcalSCVetoFromClusts"),
        weight = cms.string('1'),
        deltaR = cms.double(0.4),
        vetos = cms.vstring(),
        skipDefaultVeto = cms.bool(True)
    ))
)

eleIsoFromDepsEcalFromClusts = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        mode = cms.string('sum'),
        src = cms.InputTag("eleIsoDepositEcalFromClusts"),
        weight = cms.string('1'),
        deltaR = cms.double(0.4),
        vetos = cms.vstring('EcalBarrel:0.045', 
                            'EcalBarrel:RectangularEtaPhiVeto(-0.01,0.01,-0.5,0.5)', 
                            'EcalEndcaps:0.070', 
                            'EcalEndcaps:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)'),
        skipDefaultVeto = cms.bool(True)
    ))
)

eleIsoFromDepsEcalFromHits = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        mode = cms.string('sum'),
        src = cms.InputTag("eleIsoDepositEcalFromHits"),
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

eleIsoFromDepsHcalFromHits = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        mode = cms.string('sum'),
        src = cms.InputTag("eleIsoDepositHcalFromHits"),
        weight = cms.string('1'),
        deltaR = cms.double(0.4),
        vetos = cms.vstring('0.0'),
        skipDefaultVeto = cms.bool(True)
    ))
)

eleIsoFromDepsHcalFromTowers = cms.EDFilter("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("eleIsoDepositHcalFromTowers"),
        deltaR = cms.double(0.4),
        weight = cms.string('1'),
        vetos = cms.vstring('0.00'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)


