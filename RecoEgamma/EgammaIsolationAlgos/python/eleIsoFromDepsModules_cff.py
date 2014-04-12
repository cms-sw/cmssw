import FWCore.ParameterSet.Config as cms

eleIsoFromDepsTk = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        mode = cms.string('sum'),
        src = cms.InputTag("eleIsoDepositTk"),
        weight = cms.string('1'),
        deltaR = cms.double(0.3),
        vetos = cms.vstring('RectangularEtaPhiVeto(-0.015,0.015,-0.5,0.5)','Threshold(0.7)'),
        skipDefaultVeto = cms.bool(True)
    ))
)

eleIsoFromDepsEcalFromHits= cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        mode = cms.string('sum'),
        src = cms.InputTag("eleIsoDepositEcalFromHits"),
        weight = cms.string('1'),
        deltaR = cms.double(0.4),
        vetos = cms.vstring('EcalBarrel:0.045', 
                            'EcalBarrel:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)',
                            'EcalBarrel:ThresholdFromTransverse(0.095)',
                            'EcalEndcaps:Threshold(0.110)',
                            #'EcalEndcaps:0.070', 
                            'EcalEndcaps:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)'), 
        skipDefaultVeto = cms.bool(True)
    ))
)

eleIsoFromDepsEcalFromHitsByCrystal = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        mode = cms.string('sum'),
        src = cms.InputTag("eleIsoDepositEcalFromHits"),
        weight = cms.string('1'),
        deltaR = cms.double(0.4),
        vetos = cms.vstring('NumCrystalVeto(3.0)', 
                            'NumCrystalEtaPhiVeto(1.5,9999.0)',
                            'EcalBarrel:ThresholdFromTransverse(0.095)',
                            'EcalEndcaps:Threshold(0.110)'),
        skipDefaultVeto = cms.bool(True)
    ))
)

eleIsoFromDepsHcalFromHits = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        mode = cms.string('sum'),
        src = cms.InputTag("eleIsoDepositHcalFromHits"),
        weight = cms.string('1'),
        deltaR = cms.double(0.4),
        vetos = cms.vstring('0.15'),
        skipDefaultVeto = cms.bool(True)
    ))
)

eleIsoFromDepsHcalFromTowers = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("eleIsoDepositHcalFromTowers"),
        deltaR = cms.double(0.4),
        weight = cms.string('1'),
        vetos = cms.vstring('0.15'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)

eleIsoFromDepsHcalDepth1FromTowers = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("eleIsoDepositHcalDepth1FromTowers"), #the input isodeposits
        deltaR = cms.double(0.4),
        weight = cms.string('1'),
        vetos = cms.vstring('0.15'), 
        skipDefaultVeto = cms.bool(True), 
        mode = cms.string('sum') #sum the Ets
    ))
)

eleIsoFromDepsHcalDepth2FromTowers = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(cms.PSet(
        src = cms.InputTag("eleIsoDepositHcalDepth2FromTowers"),
        deltaR = cms.double(0.4),
        weight = cms.string('1'),
        vetos = cms.vstring('0.15'),
        skipDefaultVeto = cms.bool(True),
        mode = cms.string('sum')
    ))
)
