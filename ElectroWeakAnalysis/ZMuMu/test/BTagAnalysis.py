

import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("ZToMuMuAnalysis")

process.include("FWCore/MessageLogger/data/MessageLogger.cfi")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(17810)
    input = cms.untracked.int32(100)
)

#process.load("ElectroWeakAnalysis/ZMuMu/OCTSUBSKIM_cff")

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
"file:/afs/cern.ch/user/d/degrutto/scratch0/testZmm/CMSSW_3_5_7/src/ElectroWeakAnalysis/ZMuMu/test/ZMuMuSubskim_135149.root"
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX_7TeV/zmm/testZMuMuSubSkim_1.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX_7TeV/zmm/testZMuMuSubSkim_2.root",
#    "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX_7TeV/wmn/testZMuMuSubSkim_1.root",
#    "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX_7TeV/wmn/testZMuMuSubSkim_2.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX_7TeV/wmn/testZMuMuSubSkim_3.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX_7TeV/wmn/testZMuMuSubSkim_4.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX_7TeV/wmn/testZMuMuSubSkim_5.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX_7TeV/wmn/testZMuMuSubSkim_6.root",
#"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX_7TeV/TTbar/testZMuMuSubSkim_1.root",
    )
)



process.TFileService = cms.Service(
        "TFileService",
            fileName = cms.string("ewkZMuMuCategories_oneshot_all_3_Test.root")
        )




process.globalMuQualityCutsAnalysis= cms.EDAnalyzer(
    "BjetAnalysis",
# actually one can clean all it up.....  I don't need any other branch..... 
    src = cms.InputTag("muons"), # dimuonsOneTrack, dimuonsOneStandAlone
    ptMin = cms.untracked.double(0.0),
    massMin = cms.untracked.double(0.0),
    massMax = cms.untracked.double(120.0),
    etaMin = cms.untracked.double(-1.0),
    etaMax = cms.untracked.double(10.0),
    trkIso = cms.untracked.double(10000),
    chi2Cut = cms.untracked.double(10),
    nHitCut = cms.untracked.int32(10)
 )



process.initialGoodZToMuMuPath = cms.Path( 
    process.globalMuQualityCutsAnalysis
)

  
#process.endPath = cms.EndPath( 
#    process.out
#)

