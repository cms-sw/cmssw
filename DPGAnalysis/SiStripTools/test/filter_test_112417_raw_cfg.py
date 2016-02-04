import FWCore.ParameterSet.Config as cms

process = cms.Process("RAWFilterTest")

process.load("DPGAnalysis.SiStripTools.filtertest.config_112417_raw_cff")

process.eventtimedistribfilter = process.eventtimedistribution.clone()

#------------------------------------------------------------------
# filters
#------------------------------------------------------------------
process.load("DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_HugeEvents_AlCaReco_cfi")
process.PotentialTIBTECHugeEvents.commonConfiguration.historyProduct = cms.untracked.InputTag("consecutiveHEs")
process.PotentialTIBTECHugeEvents.commonConfiguration.APVPhaseLabel = cms.untracked.string("APVPhases")
#------------------------------------------------------------------


process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('filter_test_112417_raw.root')
                                   )

process.p0 = cms.Path(process.sinit +
                      process.eventtimedistribution +
                      process.PotentialTIBTECHugeEvents +
                      process.eventtimedistribfilter )

