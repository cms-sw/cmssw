import FWCore.ParameterSet.Config as cms

TimingFillerBlock = cms.PSet(
    TimingFillerParameters = cms.PSet(
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
        ),
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
        ),
        EcalEnergyCut = cms.double(0.4),
        ErrorEB = cms.double(2.085),
        ErrorEE = cms.double(6.95),
        MatchParameters = cms.PSet(
            CSCsegments = cms.InputTag("cscSegments"),
            DTradius = cms.double(0.01),
            DTsegments = cms.InputTag("dt4DSegments"),
            RPChits = cms.InputTag("rpcRecHits"),
            TightMatchCSC = cms.bool(True),
            TightMatchDT = cms.bool(False)
        ),
        UseCSC = cms.bool(True),
        UseDT = cms.bool(True),
        UseECAL = cms.bool(False)
    )
)