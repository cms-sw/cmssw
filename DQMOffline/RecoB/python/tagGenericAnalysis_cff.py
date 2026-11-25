import FWCore.ParameterSet.Config as cms

bTagGenericAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        discriminatorStart = cms.double(-0.01),
        nBinEffPur = cms.int32(200),
        # the constant b-efficiency for the differential plots versus pt and eta
        effBConst = cms.double(0.7),
        endEffPur = cms.double(1.005),
        discriminatorEnd = cms.double(1.011),
        startEffPur = cms.double(0.005)
    )
)

cTagGenericAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        discriminatorStart = cms.double(-0.01),
        nBinEffPur = cms.int32(200),
        # the constant c-efficiency for the differential plots versus pt and eta
        effBConst = cms.double(0.7),
        endEffPur = cms.double(1.005),
        discriminatorEnd = cms.double(1.011),
        startEffPur = cms.double(0.005)
    )
)

tauTagGenericAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        discriminatorStart = cms.double(-0.01),
        nBinEffPur = cms.int32(200),
        # the constant tau-efficiency for the differential plots versus pt and eta
        effBConst = cms.double(0.7),
        endEffPur = cms.double(1.005),
        discriminatorEnd = cms.double(1.011),
        startEffPur = cms.double(0.005)
    )
)

sTagGenericAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        discriminatorStart = cms.double(-0.01),
        nBinEffPur = cms.int32(200),
        # the constant s-efficiency for the differential plots versus pt and eta
        effBConst = cms.double(0.7),
        endEffPur = cms.double(1.005),
        discriminatorEnd = cms.double(1.011),
        startEffPur = cms.double(0.005)
    )
)

qgTagGenericAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        discriminatorStart = cms.double(-0.01),
        nBinEffPur = cms.int32(200),
        # the constant q-efficiency for the differential plots versus pt and eta
        effBConst = cms.double(0.7),
        endEffPur = cms.double(1.005),
        discriminatorEnd = cms.double(1.011),
        startEffPur = cms.double(0.005)
    )
)

# TrackProbability jetTag configuration
bTagProbabilityAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        discriminatorStart = cms.double(-0.025),
        nBinEffPur = cms.int32(200),
        # the constant b-efficiency for the differential plots versus pt and eta
        effBConst = cms.double(0.5),
        endEffPur = cms.double(1.005),
        discriminatorEnd = cms.double(2.525),
        startEffPur = cms.double(0.005)
    )
)

# TrackIP tag info configuration
bTagTrackIPAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        QualityPlots = cms.bool(False),
        endEffPur = cms.double(1.005),
        nBinEffPur = cms.int32(200),
        startEffPur = cms.double(0.005),
        LowerIPSBound = cms.double(-35.0),
        UpperIPSBound = cms.double(35.0),
        LowerIPBound = cms.double(-0.1),
        UpperIPBound = cms.double(0.1),
        LowerIPEBound = cms.double(0.0),
        UpperIPEBound = cms.double(0.04),
        NBinsIPS = cms.int32(100),
        NBinsIP = cms.int32(100),
        NBinsIPE = cms.int32(100),
        # IP histograms with more than 2 % outside the range above on the 20_1_0_pre2 RelVals (TTbar and QCD
        # PU, BTagMu 2025E, Phase-2 TTbar PU200): booked on their own range with the same bin width
        WideRangeHistograms = cms.VPSet(
            cms.PSet(name = cms.string('ip_3D'), lower = cms.double(-0.92), upper = cms.double(0.66)),
            cms.PSet(name = cms.string('ip1_3D'), lower = cms.double(-0.7), upper = cms.double(1.04)),
            cms.PSet(name = cms.string('ip2_3D'), lower = cms.double(-1.36), upper = cms.double(0.3)),
            cms.PSet(name = cms.string('ip3_3D'), lower = cms.double(-0.44), upper = cms.double(0.3)),
            cms.PSet(name = cms.string('ip4_3D'), lower = cms.double(-0.44), upper = cms.double(0.2)),
            cms.PSet(name = cms.string('ip_2D'), lower = cms.double(-0.3), upper = cms.double(0.3)),
            cms.PSet(name = cms.string('ip1_2D'), lower = cms.double(-0.3), upper = cms.double(0.3)),
            cms.PSet(name = cms.string('ip2_2D'), lower = cms.double(-0.3), upper = cms.double(0.3)),
            cms.PSet(name = cms.string('ips_3D'), lower = cms.double(-231.0), upper = cms.double(196.0)),
            cms.PSet(name = cms.string('ips1_3D'), lower = cms.double(-105.0), upper = cms.double(280.0)),
            cms.PSet(name = cms.string('ips2_3D'), lower = cms.double(-252.0), upper = cms.double(119.0)),
            cms.PSet(name = cms.string('ips3_3D'), lower = cms.double(-98.0), upper = cms.double(70.0)),
            cms.PSet(name = cms.string('ips4_3D'), lower = cms.double(-112.0), upper = cms.double(56.0)),
            cms.PSet(name = cms.string('ips_2D'), lower = cms.double(-35.0), upper = cms.double(105.0)),
            cms.PSet(name = cms.string('ips1_2D'), lower = cms.double(-105.0), upper = cms.double(175.0)),
            cms.PSet(name = cms.string('ips2_2D'), lower = cms.double(-35.0), upper = cms.double(105.0)),
            cms.PSet(name = cms.string('ips3_2D'), lower = cms.double(-35.0), upper = cms.double(63.0)),
        ),
        MinDecayLength = cms.double(-9999.0),
        MaxDecayLength = cms.double(5.0),
        MinJetDistance = cms.double(0.0),
        MaxJetDistance = cms.double(0.07),
    )
)
