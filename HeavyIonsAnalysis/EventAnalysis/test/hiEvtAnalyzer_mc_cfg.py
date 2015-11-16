import FWCore.ParameterSet.Config as cms

process = cms.Process('EvtAna')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
			    fileNames = cms.untracked.vstring('root://xrootd.unl.edu//store/user/twang/Hydjet_Quenched_MinBias_5020GeV_750/Hydjet_Quenched_MinBias_5020GeV_750_HiFall15_step3_20151110/8279ae7c7b9873cb2e7129d3c6d86a22/step3_RAW2DIGI_L1Reco_RECO_1000_1_Qj2.root'),
)

process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(-1))

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag, '75X_mcRun2_HeavyIon_v8', '')

process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")

process.TFileService = cms.Service("TFileService",
                                  fileName=cms.string("eventtree_mc.root"))

process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_mc_cfi')
process.hiEvtAnalyzer.doMC          = cms.bool(False) # because heavyIon gen event is not yet in AOD

process.p = cms.Path(process.centralityBin * process.hiEvtAnalyzer)
