import FWCore.ParameterSet.Config as cms

# following thresholds are optimized using a sample of single muons 
# with flat pT (10 - 50 GeV) and eta (0.8 - 2.4) distributions
#
# The cuts dRPhiFineMax and dPhiFineMax where re-optimized using
# MTCC data.
TC_ME1234 = cms.PSet(
    dPhiFineMax = cms.double(0.02),
    verboseInfo = cms.untracked.bool(True),
    SegmentSorting = cms.int32(1),
    chi2Max = cms.double(6000.0),
    dPhiMax = cms.double(0.003),
    chi2ndfProbMin = cms.double(0.0001),
    minLayersApart = cms.int32(2),
    dRPhiFineMax = cms.double(6.0),
    dRPhiMax = cms.double(1.2)
)
TC_ME1A = cms.PSet(
    dPhiFineMax = cms.double(0.013),
    verboseInfo = cms.untracked.bool(True),
    SegmentSorting = cms.int32(1),
    chi2Max = cms.double(6000.0),
    dPhiMax = cms.double(0.00198),
    chi2ndfProbMin = cms.double(0.0001),
    minLayersApart = cms.int32(2),
    dRPhiFineMax = cms.double(3.0),
    dRPhiMax = cms.double(0.6)
)
CSCSegAlgoTC = cms.PSet(
    chamber_types = cms.vstring('ME1/a', 'ME1/b', 'ME1/2', 'ME1/3', 'ME2/1', 'ME2/2', 'ME3/1', 'ME3/2', 'ME4/1','ME4/2'),
    algo_name = cms.string('CSCSegAlgoTC'),
    algo_psets = cms.VPSet( cms.PSet(TC_ME1234), cms.PSet(TC_ME1A) ),
    parameters_per_chamber_type = cms.vint32( 2, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
)

