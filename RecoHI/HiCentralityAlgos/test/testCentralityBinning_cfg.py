
import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

#process.load("RecoHI.HiCentralityAlgos.HiTrivialCondRetriever_cfi")

process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:/net/hisrv0001/home/yutingb/work/Centrality_3_7_0/src/RecoHI/HiCentralityAlgos/data/CentralityTables.db'

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                       process.CondDBCommon,
                                       DumpStat=cms.untracked.bool(True),
                                       toGet = cms.VPSet(cms.PSet(
                                               record = cms.string('HeavyIonRcd'),
                                               tag = cms.string('HFhits40_MC_Hydjet2760GeV_MC_3XY_V24_v0')
                                                )),
                                      )

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
        )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("rfio:/castor/cern.ch/user/y/yilmaz/share/Hydjet_MinBias_Jets_runs1to20.root")
                            )

process.analyze = cms.EDAnalyzer("AnalyzerWithCentrality")

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string("histograms.root")
                                   )

process.p = cms.Path(process.analyze)

