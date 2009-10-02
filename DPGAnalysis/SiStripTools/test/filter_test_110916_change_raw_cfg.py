import FWCore.ParameterSet.Config as cms

process = cms.Process("RAWFilterTest")

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
from DPGAnalysis.SiStripTools.filtertest.raw_110916_change_cff import fileNames
process.source.fileNames = fileNames

#---------------------------------------------------------------------
# Raw to Digi: TO BE TESTED !!
#---------------------------------------------------------------------
process.load("CondCore.DBCommon.CondDBSetup_cfi")

# Magnetic fiuld: force mag field to be 3.8 tesla
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#Geometry
process.load("Configuration.StandardSequences.Geometry_cff")

# Real data raw to digi
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

#-------------------------------------------------
# Global Tag
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR09_31X_V5P::All"


#-------------------------------------------------------------------------

process.consecutiveHEs = cms.EDProducer("EventWithHistoryProducerFromL1ABC",
                                       l1ABCCollection=cms.InputTag("scalersRawToDigi")
                                       )


#------------------------------------------------------------------------
# APV Cycle Phase Producer and monitor
#------------------------------------------------------------------------
process.load("DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1abc_GR09_cfi")

#------------------------------------------------------------------------

process.load("DPGAnalysis.SiStripTools.eventtimedistribution_cfi")
process.eventtimedistribfilter = process.eventtimedistribution.clone()

process.load("DPGAnalysis.SiStripTools.apvlatency.fakeapvlatencyessource_cff")
process.fakeapvlatency.APVLatency = cms.untracked.int32(144)

#------------------------------------------------------------------
# filters
#------------------------------------------------------------------
process.load("DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_HugeEvents_AlCaReco_cfi")
process.PotentialTIBTECHugeEvents.commonConfiguration.historyProduct = cms.untracked.InputTag("consecutiveHEs")
process.PotentialTIBTECHugeEvents.commonConfiguration.APVPhaseLabel = cms.untracked.string("APVPhases")
#------------------------------------------------------------------


process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('filter_test_110916_change_raw.root')
                                   )

process.p0 = cms.Path(process.scalersRawToDigi +
                      process.consecutiveHEs + process.APVPhases +
                      process.eventtimedistribution +
                      process.PotentialTIBTECHugeEvents +
                      process.eventtimedistribfilter )

