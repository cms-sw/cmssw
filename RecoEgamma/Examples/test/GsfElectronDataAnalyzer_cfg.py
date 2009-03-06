import FWCore.ParameterSet.Config as cms

process = cms.Process("readelectrons")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring (
    '/store/relval/2008/7/21/RelVal-RelValSingleElectronPt35-1216579481-IDEAL_V5-2nd/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/0808CDA5-6E57-DD11-9CC1-000423D99896.root',
    '/store/relval/2008/7/21/RelVal-RelValSingleElectronPt35-1216579481-IDEAL_V5-2nd/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/1AEAE04D-6E57-DD11-AAA9-000423D94AA8.root',
    '/store/relval/2008/7/21/RelVal-RelValSingleElectronPt35-1216579481-IDEAL_V5-2nd/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/B6322A11-6E57-DD11-8A69-000423D99264.root'
    )
)

process.mergedSuperClusters = cms.EDFilter("EgammaSuperClusterMerger",
    src = cms.VInputTag(cms.InputTag("correctedHybridSuperClusters"), cms.InputTag("multi5x5SuperClustersWithPreshower"))
)

process.gsfElectronAnalysis = cms.EDAnalyzer("GsfElectronDataAnalyzer",
    electronCollection = cms.InputTag("pixelMatchGsfElectrons"),
    Nbinxyz = cms.int32(50),
    Nbineop2D = cms.int32(30),
    Nbinp = cms.int32(50),
    Lhitsmax = cms.double(10.0),
    Etamin = cms.double(-2.5),
    Nbinfhits = cms.int32(20),
    Eopmax = cms.double(5.0),
    Pmax = cms.double(300.0),
    Phimax = cms.double(3.2),
    Phimin = cms.double(-3.2),
    Dphimin = cms.double(-0.15),
    MaxPt = cms.double(100.0),
    Nbinlhits = cms.int32(5),
    Nbinpteff = cms.int32(19),
    Nbinphi2D = cms.int32(32),
    Nbindetamatch2D = cms.int32(50),
    Nbineta = cms.int32(50),
    DeltaR = cms.double(0.3),
    outputFile = cms.string('gsfElectronHistos_data.root'),
    Nbinp2D = cms.int32(50),
    Nbindeta = cms.int32(100),
    Nbinpt2D = cms.int32(50),
    Nbindetamatch = cms.int32(100),
    Fhitsmax = cms.double(20.0),
    matchingObjectCollection = cms.InputTag("mergedSuperClusters"),
    Nbinphi = cms.int32(64),
    Nbineta2D = cms.int32(50),
    Eopmaxsht = cms.double(3.0),
    MaxAbsEta = cms.double(2.5),
    Nbindphimatch = cms.int32(100),
    Detamax = cms.double(0.15),
    Nbinpt = cms.int32(50),
    Nbindphimatch2D = cms.int32(50),
    Etamax = cms.double(2.5),
    Dphimax = cms.double(0.15),
    Dphimatchmax = cms.double(0.2),
    Detamatchmax = cms.double(0.05),
    Nbindphi = cms.int32(100),
    Detamatchmin = cms.double(-0.05),
    Ptmax = cms.double(100.0),
    Nbineop = cms.int32(50),
    Dphimatchmin = cms.double(-0.2),
    Detamin = cms.double(-0.15)
)

process.p = cms.Path(process.mergedSuperClusters*process.gsfElectronAnalysis)


