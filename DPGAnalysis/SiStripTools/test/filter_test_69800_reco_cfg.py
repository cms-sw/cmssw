import FWCore.ParameterSet.Config as cms

process = cms.Process("RECOFilterTest")

process.load("DPGAnalysis.SiStripTools.filtertest.config_69800_reco_cff")

process.eventtimedistribfilter = process.eventtimedistribution.clone()

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

