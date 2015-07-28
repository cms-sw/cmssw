import FWCore.ParameterSet.Config as cms

process = cms.Process('EvtAna')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
			    fileNames = cms.untracked.vstring('root://xrootd.unl.edu//store/user/tuos/HIAOD2015/round3/June01/HydjetMBRECO/Hydjet_Quenched_MinBias_5020GeV/HydjetMB_RECO_750pre5_round3v01/150618_204846/0000/step2_RAW2DIGI_L1Reco_MB_RECOSIM_1.root'),
)

process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(-1))

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag, '75X_mcRun2_HeavyIon_v2', '')
process.GlobalTag.toGet.extend([
   cms.PSet(record = cms.string("HeavyIonRcd"),
      tag = cms.string("CentralityTable_HFtowers200_HydjetDrum5_v750x02_mc"),
      connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
      label = cms.untracked.string("HFtowersHydjetDrum5")
   ),
])

process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi") 
process.centralityBin.Centrality = cms.InputTag("hiCentrality")
process.centralityBin.centralityVariable = cms.string("HFtowers")
process.centralityBin.nonDefaultGlauberModel = cms.string("HydjetDrum5")

process.TFileService = cms.Service("TFileService",
                                  fileName=cms.string("eventtree_mc.root"))

process.load('GeneratorInterface.HiGenCommon.HeavyIon_cff') #because of this it only runs on RECO now

process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_mc_cfi')

process.p = cms.Path(process.heavyIon * process.centralityBin * process.hiEvtAnalyzer)
