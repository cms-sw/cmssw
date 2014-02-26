
import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTC")

process.HeavyIonGlobalParameters = cms.PSet(
    centralitySrc = cms.InputTag("hiCentrality"),
    centralityVariable = cms.string("HFhits"),
    nonDefaultGlauberModel = cms.string("")
    )

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR_R_39X_V1::All'

from CmsHi.Analysis2010.CommonFunctions_cff import *
overrideCentrality(process)

process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string("runnumber"),
                            firstValue = cms.uint64(151076),
                            lastValue = cms.uint64(151076),
                            interval = cms.uint64(1)
                            )

process.analyze = cms.EDAnalyzer("AnalyzerWithCentrality")

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string("histogramsAndTable.root")
                                   )

process.p = cms.Path(process.analyze)

