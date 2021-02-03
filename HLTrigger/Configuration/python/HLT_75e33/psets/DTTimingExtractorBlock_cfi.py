import FWCore.ParameterSet.Config as cms

DTTimingExtractorBlock = cms.PSet(
    DTTimingParameters = cms.PSet(
        DTTimeOffset = cms.double(0.0),
        DoWireCorr = cms.bool(True),
        DropTheta = cms.bool(True),
        HitError = cms.double(2.8),
        HitsMin = cms.int32(3),
        PruneCut = cms.double(5.0),
        RequireBothProjections = cms.bool(False),
        ServiceParameters = cms.PSet(
            Propagators = cms.untracked.vstring(
                'SteppingHelixPropagatorAny',
                'PropagatorWithMaterial',
                'PropagatorWithMaterialOpposite'
            ),
            RPCLayers = cms.bool(True)
        ),
        UseSegmentT0 = cms.bool(False),
        debug = cms.bool(False)
    )
)