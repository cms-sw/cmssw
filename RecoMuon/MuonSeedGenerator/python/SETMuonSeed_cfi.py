import FWCore.ParameterSet.Config as cms

# The services
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
# parametrization for initial pT
from RecoMuon.MuonSeedGenerator.ptSeedParameterization_38T_cfi import ptSeedParameterization
from RecoMuon.MuonSeedGenerator.MuonSeedPtScale_cfi import dphiScale

SETMuonSeed  = cms.EDProducer("SETMuonSeedProducer",
    MuonServiceProxy,
    SETTrajBuilderParameters = cms.PSet(
        ptSeedParameterization, 
        dphiScale,
        Apply_prePruning = cms.bool(True),
# Careful - next is used together with useSubRecHits in standAloneMuons_cfi.py (for now)
#	UseSegmentsInTrajectory = cms.bool(True),
	UseSegmentsInTrajectory = cms.bool(False),
        FilterParameters = cms.PSet(
            DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
            CSCRecSegmentLabel = cms.InputTag("cscSegments"),
            RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),
            Propagator = cms.string('SteppingHelixPropagatorAny'),

            EnableRPCMeasurement = cms.bool(True),
# NOT USED for now
            EnableDTMeasurement = cms.bool(True),
            EnableCSCMeasurement = cms.bool(True)       
        )
    )
)



