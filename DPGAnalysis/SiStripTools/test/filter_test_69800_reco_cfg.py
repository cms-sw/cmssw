import FWCore.ParameterSet.Config as cms

process = cms.Process("RECOFilterTest")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.categories.append("L1AcceptBunchCrossingNoCollection")
process.MessageLogger.categories.append("EventWithHistoryFilterConfiguration")

process.MessageLogger.infos.placeholder = cms.untracked.bool(False)
process.MessageLogger.infos.threshold = cms.untracked.string("INFO")
process.MessageLogger.infos.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.infos.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10000)
    )
process.MessageLogger.infos.L1AcceptBunchCrossingNoCollection = cms.untracked.PSet(
    limit = cms.untracked.int32(100)
    )
process.MessageLogger.cerr.L1AcceptBunchCrossingNoCollection = cms.untracked.PSet(
    limit = cms.untracked.int32(100)
    )
#process.MessageLogger.infos.EventWithHistoryFilterConfiguration = cms.untracked.PSet(
#    limit = cms.untracked.int32(0)
#    )
process.MessageLogger.cerr.threshold = cms.untracked.string("WARNING")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(),
                            skipBadFiles = cms.untracked.bool(True)
                            )
from DPGAnalysis.SiStripTools.filtertest.reco_69800_debug_cff import fileNames
process.source.fileNames = fileNames

#-------------------------------------------------------------------------

process.load("DPGAnalysis.SiStripTools.eventwithhistoryproducer_cfi")

#------------------------------------------------------------------------
# APV Cycle Phase Producer and monitor
#------------------------------------------------------------------------
process.load("DPGAnalysis.SiStripTools.configurableapvcyclephaseproducer_CRAFT08_cfi")

#------------------------------------------------------------------------

process.load("DPGAnalysis.SiStripTools.eventtimedistribution_cfi")
process.eventtimedistribfilter = process.eventtimedistribution.clone()

process.load("DPGAnalysis.SiStripTools.apvlatency.fakeapvlatencyessource_cff")
process.fakeapvlatency.APVLatency = cms.untracked.int32(143)

#------------------------------------------------------------------
# filters
#------------------------------------------------------------------
process.load("DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_HugeEvents_AlCaReco_cfi")
process.PotentialTIBTECHugeEvents.commonConfiguration.historyProduct = cms.untracked.InputTag("consecutiveHEs")
process.PotentialTIBTECHugeEvents.commonConfiguration.APVPhaseLabel = cms.untracked.string("APVPhases")
#------------------------------------------------------------------


process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('filter_test_69800_reco.root')
                                   )

process.p0 = cms.Path(
                      process.consecutiveHEs + process.APVPhases +
                      process.eventtimedistribution +
                      process.PotentialTIBTECHugeEvents +
                      process.eventtimedistribfilter )

