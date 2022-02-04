import FWCore.ParameterSet.Config as cms

hltEgammaPixelMatchVarsL1Seeded = cms.EDProducer("EgammaHLTPixelMatchVarProducer",
    dPhi1SParams = cms.PSet(
        bins = cms.VPSet(
            cms.PSet(
                binType = cms.string('AbsEtaClus'),
                funcParams = cms.vdouble(0.00273931, -0.00251994, 0.00324979),
                funcType = cms.string('TF1:=pol2'),
                xMax = cms.double(1.5),
                xMin = cms.double(0.0),
                yMax = cms.int32(1),
                yMin = cms.int32(1)
            ),
            cms.PSet(
                binType = cms.string('AbsEtaClus'),
                funcParams = cms.vdouble(0.00465536, -0.00170883, 0.0022395),
                funcType = cms.string('TF1:=pol2'),
                xMax = cms.double(1.5),
                xMin = cms.double(0.0),
                yMax = cms.int32(2),
                yMin = cms.int32(2)
            ),
            cms.PSet(
                binType = cms.string('AbsEtaClus'),
                funcParams = cms.vdouble(0.00612202, -0.000985677, 0.00230772),
                funcType = cms.string('TF1:=pol2'),
                xMax = cms.double(1.5),
                xMin = cms.double(0.0),
                yMax = cms.int32(99999),
                yMin = cms.int32(3)
            ),
            cms.PSet(
                binType = cms.string('AbsEtaClus'),
                funcParams = cms.vdouble(0.0379945, -0.0334501, 0.00799893),
                funcType = cms.string('TF1:=pol2'),
                xMax = cms.double(2.4),
                xMin = cms.double(1.5),
                yMax = cms.int32(1),
                yMin = cms.int32(1)
            ),
            cms.PSet(
                binType = cms.string('AbsEtaClus'),
                funcParams = cms.vdouble(0.00579179, -0.00956301, 0.00357333),
                funcType = cms.string('TF1:=pol2'),
                xMax = cms.double(4.0),
                xMin = cms.double(2.4),
                yMax = cms.int32(1),
                yMin = cms.int32(1)
            ),
            cms.PSet(
                binType = cms.string('AbsEtaClus'),
                funcParams = cms.vdouble(0.0294649, -0.0235045, 0.00566937),
                funcType = cms.string('TF1:=pol2'),
                xMax = cms.double(4.0),
                xMin = cms.double(1.5),
                yMax = cms.int32(2),
                yMin = cms.int32(2)
            ),
            cms.PSet(
                binType = cms.string('AbsEtaClus'),
                funcParams = cms.vdouble(0.0227801, -0.00899003),
                funcType = cms.string('TF1:=pol1'),
                xMax = cms.double(2.0),
                xMin = cms.double(1.5),
                yMax = cms.int32(99999),
                yMin = cms.int32(3)
            ),
            cms.PSet(
                binType = cms.string('AbsEtaClus'),
                funcParams = cms.vdouble(-0.0448686, 0.0405059, -0.00789926),
                funcType = cms.string('TF1:=pol2'),
                xMax = cms.double(4.0),
                xMin = cms.double(2.0),
                yMax = cms.int32(99999),
                yMin = cms.int32(3)
            )
        )
    ),
    dPhi2SParams = cms.PSet(
        bins = cms.VPSet(
            cms.PSet(
                binType = cms.string('AbsEtaClus'),
                funcParams = cms.vdouble(0.000262924, -0.00012575),
                funcType = cms.string('TF1:=pol1'),
                xMax = cms.double(0.6),
                xMin = cms.double(0.0),
                yMax = cms.int32(99999),
                yMin = cms.int32(1)
            ),
            cms.PSet(
                binType = cms.string('AbsEtaClus'),
                funcParams = cms.vdouble(-0.000283732, 0.00105965, -0.000460304),
                funcType = cms.string('TF1:=pol2'),
                xMax = cms.double(1.47),
                xMin = cms.double(0.6),
                yMax = cms.int32(99999),
                yMin = cms.int32(1)
            ),
            cms.PSet(
                binType = cms.string('AbsEtaClus'),
                funcParams = cms.vdouble(0.00172122, 0.00149787, 0.000370645),
                funcType = cms.string('TF1:=pol2'),
                xMax = cms.double(4.0),
                xMin = cms.double(1.47),
                yMax = cms.int32(99999),
                yMin = cms.int32(1)
            )
        )
    ),
    dRZ2SParams = cms.PSet(
        bins = cms.VPSet(
            cms.PSet(
                binType = cms.string('AbsEtaClus'),
                funcParams = cms.vdouble(0.00502445, -0.0047799, 0.00808078),
                funcType = cms.string('TF1:=pol2'),
                xMax = cms.double(1.13),
                xMin = cms.double(0.0),
                yMax = cms.int32(99999),
                yMin = cms.int32(1)
            ),
            cms.PSet(
                binType = cms.string('AbsEtaClus'),
                funcParams = cms.vdouble(0.2007, -0.305712, 0.121756),
                funcType = cms.string('TF1:=pol2'),
                xMax = cms.double(1.48),
                xMin = cms.double(1.13),
                yMax = cms.int32(99999),
                yMin = cms.int32(1)
            ),
            cms.PSet(
                binType = cms.string('AbsEtaClus'),
                funcParams = cms.vdouble(0.169387, -0.177821, 0.0477192),
                funcType = cms.string('TF1:=pol2'),
                xMax = cms.double(1.9),
                xMin = cms.double(1.48),
                yMax = cms.int32(99999),
                yMin = cms.int32(1)
            ),
            cms.PSet(
                binType = cms.string('AbsEtaClus'),
                funcParams = cms.vdouble(0.0245799, -0.0197369, 0.00451283),
                funcType = cms.string('TF1:=pol2'),
                xMax = cms.double(4.0),
                xMin = cms.double(1.9),
                yMax = cms.int32(99999),
                yMin = cms.int32(1)
            )
        )
    ),
    pixelSeedsProducer = cms.InputTag("hltEgammaElectronPixelSeedsL1Seeded"),
    productsToWrite = cms.int32(0),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesL1Seeded")
)
