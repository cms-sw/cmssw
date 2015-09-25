import FWCore.ParameterSet.Config as cms

process = cms.Process('EvtAna')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
			    fileNames = cms.untracked.vstring('root://xrootd.unl.edu//store/user/tuos/HIAOD2015/round3/June01/MB/HIMinBiasUPC/MB_AOD_DATA750pre5_round3v01/150601_201116/0000/step2_RAW2DIGI_L1Reco_MB_AOD_1.root'),
)

process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(-1))

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run1_data', '')
process.GlobalTag.snapshotTime = cms.string("9999-12-31 23:59:59.000")
process.GlobalTag.toGet.extend([
   cms.PSet(record = cms.string("HeavyIonRcd"),
      tag = cms.string("CentralityTable_HFtowers200_Glauber2010A_eff99_run1v750x01_offline"),
      connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
      label = cms.untracked.string("HFtowers")
   ),
])

process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi") 
process.centralityBin.Centrality = cms.InputTag("hiCentrality")
process.centralityBin.centralityVariable = cms.string("HFtowers")
process.centralityBin.nonDefaultGlauberModel = cms.string("")

process.TFileService = cms.Service("TFileService",
                                  fileName=cms.string("eventtree_filtered_data.root"))


process.load('HeavyIonsAnalysis.Configuration.collisionEventSelection_cff')

process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_data_cfi')

process.p = cms.Path(process.collisionEventSelectionAOD * process.centralityBin * process.hiEvtAnalyzer)
