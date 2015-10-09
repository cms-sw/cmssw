import FWCore.ParameterSet.Config as cms

# following thresholds are optimized using a sample of single muons 
# with flat pT (10 - 50 GeV) and eta (0.8 - 2.4) distributions
SK_ME1234 = cms.PSet(
    dPhiFineMax = cms.double(0.025),
    verboseInfo = cms.untracked.bool(True),
    chi2Max = cms.double(99999.0),
    dPhiMax = cms.double(0.003),
    wideSeg = cms.double(3.0),
    minLayersApart = cms.int32(2),
    dRPhiFineMax = cms.double(8.0),
    dRPhiMax = cms.double(8.0)
)
SK_ME1A = cms.PSet(
    dPhiFineMax = cms.double(0.025),
    verboseInfo = cms.untracked.bool(True),
    chi2Max = cms.double(99999.0),
    dPhiMax = cms.double(0.025),
    wideSeg = cms.double(3.0),
    minLayersApart = cms.int32(2),
    dRPhiFineMax = cms.double(3.0),
    dRPhiMax = cms.double(8.0)
)
CSCSegAlgoSK = cms.PSet(
    chamber_types = cms.vstring('ME1/a', 'ME1/b', 'ME1/2', 'ME1/3', 'ME2/1', 'ME2/2', 'ME3/1', 'ME3/2', 'ME4/1','ME4/2'),
    algo_name = cms.string('CSCSegAlgoSK'),
    algo_psets = cms.VPSet( cms.PSet(SK_ME1234), cms.PSet(SK_ME1A) ),
    parameters_per_chamber_type = cms.vint32(2, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
)

