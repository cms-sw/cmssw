import FWCore.ParameterSet.Config as cms

CSCTimingExtractorBlock = cms.PSet(
    CSCTimingParameters = cms.PSet(
        CSCStripError = cms.double(7.0),
        CSCStripTimeOffset = cms.double(0.0),
        CSCWireError = cms.double(8.6),
        CSCWireTimeOffset = cms.double(0.0),
        PruneCut = cms.double(9.0),
        ServiceParameters = cms.PSet(
            Propagators = cms.untracked.vstring(
                'SteppingHelixPropagatorAny',
                'PropagatorWithMaterial',
                'PropagatorWithMaterialOpposite'
            ),
            RPCLayers = cms.bool(True)
        ),
        UseStripTime = cms.bool(True),
        UseWireTime = cms.bool(True),
        debug = cms.bool(False)
    )
)