
import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.HeavyIonGlobalParameters = cms.PSet(
    centralityVariable = cms.string(""),
    nonDefaultGlauberModel = cms.string("AMPT_2760GeV")
    )

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START39_V4HI::All'

process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("HeavyIonRcd"),
             tag = cms.string("CentralityTable_HFhits40_Hydjet2760GeV_v1_mc"),
             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS")
             )
    )


process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
        )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("rfio:/castor/cern.ch/user/y/yilmaz/share/Hydjet_MinBias_Jets_runs1to20.root")
                            )

process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")
process.analyze = cms.EDAnalyzer("AnalyzerWithCentrality")

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string("histograms.root")
                                   )

process.p = cms.Path(process.centralityBin*process.analyze)

