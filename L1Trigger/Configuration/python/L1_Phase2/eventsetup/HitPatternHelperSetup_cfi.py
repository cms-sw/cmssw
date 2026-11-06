import FWCore.ParameterSet.Config as cms

HitPatternHelperSetup = cms.ESProducer("hph::ProducerHPH",
    hphDebug = cms.bool(False),
    oldKFPSet = cms.PSet(
        AllowOver2EtaSecs = cms.bool(True),
        BeamWindowZ = cms.double(15),
        ChosenRofZ = cms.double(50.0),
        EtaRegions = cms.vdouble(
            -2.4, -2.08, -1.68, -1.26, -0.9,
            -0.62, -0.41, -0.2, 0.0, 0.2,
            0.41, 0.62, 0.9, 1.26, 1.68,
            2.08, 2.4
        )
    ),
    useNewKF = cms.bool(False)
)
