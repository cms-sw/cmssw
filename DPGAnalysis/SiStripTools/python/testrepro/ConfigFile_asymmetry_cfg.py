import FWCore.ParameterSet.Config as cms

process = cms.Process("CRAFTRepro")

#--------------------------------------------------------------------------
# MessageLogger
#--------------------------------------------------------------------------
process.load("DPGAnalysis.SiStripTools.testrepro.MessageLogger_cff")

#-------------------------------------------------------------------
# input source
#-------------------------------------------------------------------
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1) )

process.load("DPGAnalysis.SiStripTools.testrepro.inputsource_cff")

from runfiles.RAW.raw_RUNNUMBER_cff import fileNames
process.source.fileNames = fileNames

#---------------------------------------------------------------------
# Raw to Digi: TO BE TESTED !!
#---------------------------------------------------------------------
process.load("DPGAnalysis.SiStripTools.testrepro.rawtodigi_cff")

#-------------------------------------------------
# Global Tag
#-------------------------------------------------
process.load("DPGAnalysis.SiStripTools.testrepro.globaltag_sistripquality_cff")
process.GlobalTag.globaltag = "CRAFT0831X_V1::All"

#----------------------------------------------------
# Clusterizer that uses the "unbiased" SiStripQuality
#----------------------------------------------------
process.load("DPGAnalysis.SiStripTools.testrepro.calzerobiasclusters_cff")

#-------------------------------------------------
# DQM Modules
#-------------------------------------------------
process.load("DPGAnalysis.SiStripTools.testrepro.dqm_cff")
process.dqmSaver.dirName      = 'DQMDIR'

#-------------------------------------------------------------
# Digi occupancy map and unpacker bad module map
#------------------------------------------------------------
process.load("DPGAnalysis.SiStripTools.testrepro.cs_maps_cff")
process.mapout.fileName= cms.untracked.string('selected_events_RUNNUMBER.root')
process.mapout.outputCommands = cms.untracked.vstring("keep *")

#------------------------------------------------------------------
# Filter against APV-induced noisy events
#------------------------------------------------------------------
process.load("DPGAnalysis.SiStripTools.testrepro.noisyevent_filters_2008_cff")

#------------------------------------------------------------------------
# TEC bad module analyzer
#------------------------------------------------------------------------
process.load("DPGAnalysis.SiStripTools.testrepro.tecbadmodules_cff")

#------------------------------------------------------------------------
# Selection of events with clusters in selected modules
#------------------------------------------------------------------------
process.load("DPGAnalysis.SiStripTools.sistripdetwithcluster_cfi")
process.sistripdetwithcluster.collectionName = cms.InputTag("calZeroBiasClusters")
process.sistripdetwithcluster.selectedModules = cms.untracked.vuint32(369120618,
                                                                      369120617
                                                                      )
process.load("DPGAnalysis.SiStripTools.sistripdetwithdigi_cfi")
process.sistripdetwithdigi.selectedModules = cms.untracked.vuint32(369120618)

process.pickEvents = cms.EDFilter("PickEvents",
                                  whichRun = cms.untracked.int32(69800),
                                  whichEventFirst = cms.untracked.int32(269590372),
                                  whichEventLast = cms.untracked.int32(269590372)
                                  )
#------------------------------------------------------------------------
# Digi Profiler
#------------------------------------------------------------------------
process.load("myTKAnalyses.DigiInvestigator.modulebxsliceanalyzer_cfi")
process.modulebxsliceanalyzer.moduleList = cms.untracked.vuint32(369120618)
#----------------------------------------------------------------------
# TFileService
#------------------------------------------------------------------------
process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('histos_RUNNUMBER.root')
                                   )
#------------------------------------------------------------------------


process.p = cms.Path(process.siStripDigis*process.siStripZeroSuppression*
                     process.calZeroBiasClusters*
                     process.APVPhases+
                     process.consecutiveHEs+ # APV phase and Event History producers
#filter for digi in selected modules
                     process.pickEvents+
                     process.sistripdetwithcluster+
                     process.sistripdetwithdigi+
                     process.modulebxsliceanalyzer+
#
                     process.newDigisMap+process.unpackerBadListMap+ process.unpackerBadList + #occupancy and bad modules maps
#filters to be used:
                    ~process.potentialTOBHugeEvents*
                    ~process.potentialTIBTECHugeEvents*
                    ~process.potentialTOBFrameHeaderEventsFPeak*
                    ~process.potentialTIBTECFrameHeaderEventsFPeak*
                    ~process.potentialTOBFrameHeaderEventsAdditionalPeak*
                    ~process.potentialTIBTECFrameHeaderEventsAdditionalPeak*
# DQM modules after filter
                     process.siStripFEDMonitor+
                     process.SiStripMonitorDigi+
                     process.SiStripMonitorClusterReal+
                     process.dqmSaver
# this has to be the last module !
#                     process.nooutevent
                     )

process.e = cms.EndPath(process.mapout)

