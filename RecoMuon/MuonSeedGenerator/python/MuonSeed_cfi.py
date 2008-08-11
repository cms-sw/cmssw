import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.MuonSeedGenerator.ptSeedParameterization_cfi import *
from RecoMuon.MuonSeedGenerator.MuonSeedPtScale_cfi import *
MuonSeed = cms.EDProducer("MuonSeedProducer",
    ptSeedParameterization,
    MuonServiceProxy,
    dphiScale,
    # Parameters for seed creation in overlap region
    maxDeltaEtaOverlap = cms.double(0.08),
    # Flag for internal debugging
    DebugMuonSeed = cms.bool(False),
    minimumSeedPt = cms.double(5.0),
    # The following parameters are only used in the new seed generator !
    # Parameters for seed creation in endcap region
    minCSCHitsPerSegment = cms.int32(4),
    maxDeltaPhiDT = cms.double(0.3),
    maxDeltaPhiOverlap = cms.double(0.25),
    # Parameters for seed creation in barrel region
    minDTHitsPerSegment = cms.int32(2),
    maxPhiResolutionDT = cms.double(0.03),
    DTSegmentLabel = cms.InputTag("dt4DSegments"),
    SeedPtSystematics = cms.double(0.1),
    maximumSeedPt = cms.double(3000.0),
    # Minimum and maximum Pt for seeds
    defaultSeedPt = cms.double(25.0),
    CSCSegmentLabel = cms.InputTag("cscSegments"),
    # this means 1/5 of length in ME1/a  
    maxEtaResolutionCSC = cms.double(0.06),
    # enable the DT chamber
    EnableDTMeasurement = cms.bool(True),
    # Resolution power for distinguishing between 2 muon seeds (suppression of combinatorics)
    # this means 1/20th of MB0 
    maxEtaResolutionDT = cms.double(0.02),
    maxDeltaEtaDT = cms.double(0.3),
    # this is a 5th of a chamber width
    maxPhiResolutionCSC = cms.double(0.03),
    maxDeltaEtaCSC = cms.double(0.2),
    maxDeltaPhiCSC = cms.double(0.5),
    # enable the CSC chamber
    EnableCSCMeasurement = cms.bool(True)
)



