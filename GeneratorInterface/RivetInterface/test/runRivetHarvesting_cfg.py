import FWCore.ParameterSet.Config as cms

process = cms.Process("runRivetAnalysis")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10000)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:QCD_Pt_15to3000_TuneZ2_Flat_7TeV_pythia6_cff_py_GEN.root')
)

process.load("GeneratorInterface.RivetInterface.rivetHarvesting_cfi")

process.rivetHarvesting.AnalysisNames = cms.vstring('CMS_2011_S9088458', 'CMS_2011_S8950903', 'CMS_2011_S9086218', 'CMS_FWD_10_003', 'CMS_FWD_10_006')
process.rivetHarvesting.FilesToHarvest = cms.vstring('validation_15-100.aida', 'validation_100-300.aida', 'validation_300-500.aida', 'validation_500-1000.aida', 'validation_1000-3000.aida', 'validation_3000-7000.aida')
process.rivetHarvesting.VSumOfWeights = cms.vdouble(20000.0, 20000.0, 20000.0, 20000.0, 20000.0, 20000.0)
process.rivetHarvesting.VCrossSections = cms.vdouble(877000000., 33100., 1200., 60. , 0.34, 6e-11)

process.p = cms.Path(process.rivetHarvesting)


