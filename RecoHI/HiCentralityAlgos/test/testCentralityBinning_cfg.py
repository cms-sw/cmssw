
import FWCore.ParameterSet.Config as cms

process = cms.Process("CTEST")

process.HeavyIonGlobalParameters = cms.PSet(
    centralitySrc = cms.InputTag("hiCentrality","","CTEST"),
    centralityVariable = cms.string("HFhits"),
    nonDefaultGlauberModel = cms.string("AMPT_2760GeV")
    )

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')

process.GlobalTag.globaltag = 'START39_V4::All'

process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("HeavyIonRcd"),
             tag = cms.string("CentralityTable_HFhits40_Hydjet2760GeV_v0_mc"),
             connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS")
             )
    )


process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
        )

process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring(
    "rfio:/castor/cern.ch/cms/store/caf/user/frankma/HR10Exp3/r150305HFSkim/skim_RECO_10_1_7u7.root",
),
#                            inputCommands = cms.untracked.vstring('keep *','drop *_hiCentrality_*_*')
                            )

process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")
process.analyze = cms.EDAnalyzer("AnalyzerWithCentrality")
process.hiCentrality.producePixelTracks = False
process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string("histograms.root")
                                   )

process.p = cms.Path(process.siPixelRecHits*process.hiCentrality
#                     *process.centralityBin
                     *process.analyze)

