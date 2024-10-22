import FWCore.ParameterSet.Config as cms

hltIter2Phase2L3FromL1TkMuonPixelSeedsFiltered = cms.EDProducer("MuonHLTSeedMVAClassifierPhase2",
    L1TkMu = cms.InputTag("l1tTkMuonsGmt"),
    baseScore = cms.double(0.5),
    doSort = cms.bool(True),
    etaEdge = cms.double(1.2),
    mvaCut_B = cms.double(0.0),
    mvaCut_E = cms.double(0.0),
    mvaFile_B_0 = cms.FileInPath('RecoMuon/TrackerSeedGenerator/data/xgb_Phase2_Iter2FromL1_barrel_v0.xml'),
    mvaFile_E_0 = cms.FileInPath('RecoMuon/TrackerSeedGenerator/data/xgb_Phase2_Iter2FromL1_endcap_v0.xml'),
    mvaScaleMean_B = cms.vdouble(
        0.00033113700731766336, 1.6825601468762878e-06, 1.790932122524803e-06, 0.010534608406382916, 0.005969459957330139,
        0.0009605022254971113, 0.04384189672781466, 7.846741237608237e-05, 0.40725050850004824, 0.41125151617410227,
        0.39815551065544846
    ),
    mvaScaleMean_E = cms.vdouble(
        0.00022658482374555603, 5.358921973784045e-07, 1.010003713549798e-06, 0.0007886873612224615, 0.001197730548842408,
        -0.0030252353426003594, 0.07151944804171254, -0.0006940626775109026, 0.20535152195939896, 0.2966816533783824,
        0.28798220230180455
    ),
    mvaScaleStd_B = cms.vdouble(
        0.0006042948363798624, 2.445644111872427e-06, 3.454992543447134e-06, 0.09401581628887255, 0.7978806947573766,
        0.4932933044535928, 0.04180518265631776, 0.058296511682094855, 0.4071857009373577, 0.41337782307392973,
        0.4101160349549534
    ),
    mvaScaleStd_E = cms.vdouble(
        0.0003857726789049956, 1.4853721474087994e-06, 6.982997036736564e-06, 0.04071340757666084, 0.5897606560095399,
        0.33052121398064654, 0.05589386786541949, 0.08806273533388546, 0.3254586902665612, 0.3293354496231377,
        0.3179899794578072
    ),
    nSeedsMax_B = cms.int32(20),
    nSeedsMax_E = cms.int32(20),
    src = cms.InputTag("hltIter2Phase2L3FromL1TkMuonPixelSeeds")
)
