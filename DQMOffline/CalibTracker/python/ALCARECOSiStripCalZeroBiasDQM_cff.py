import FWCore.ParameterSet.Config as cms

#------------------------
# SiStripMonitorCluster #
#------------------------

from DQM.SiStripMonitorCluster.SiStripMonitorClusterAlca_cfi import *
SiStripCalZeroBiasMonitorCluster.TopFolderName = cms.string('AlCaReco/SiStrip')

#---------------------------------------------
# Filters to remove APV induced noisy events #
#---------------------------------------------

import DPGAnalysis.SiStripTools.eventwithhistoryproducer_cfi
ConsecutiveHEs = DPGAnalysis.SiStripTools.eventwithhistoryproducer_cfi.consecutiveHEs.clone()

import DPGAnalysis.SiStripTools.configurableapvcyclephaseproducer_GR09_cfi
apvPhases = DPGAnalysis.SiStripTools.configurableapvcyclephaseproducer_GR09_cfi.APVPhases.clone()
apvPhases.defaultPhases = cms.vint32(30,30,30,30)

from DPGAnalysis.SiStripTools.apvlatency.fakeapvlatencyessource_cff import *
fakeapvlatency.APVLatency = cms.untracked.int32(144)

import DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_HugeEvents_cfi
PotentialTIBTECHugeEvents = DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_HugeEvents_cfi.potentialTIBTECHugeEvents.clone()
PotentialTIBTECHugeEvents.partitionName  = cms.untracked.string("TM")
PotentialTIBTECHugeEvents.historyProduct = cms.untracked.InputTag("ConsecutiveHEs")
PotentialTIBTECHugeEvents.APVPhaseLabel  = cms.untracked.string("apvPhases")

import DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_FrameHeaderEvents_firstpeak_cfi
PotentialTIBTECFrameHeaderEventsFPeak = DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_FrameHeaderEvents_firstpeak_cfi.potentialTIBTECFrameHeaderEventsFPeak.clone()
PotentialTIBTECFrameHeaderEventsFPeak.partitionName              = cms.untracked.string("TM")
PotentialTIBTECFrameHeaderEventsFPeak.absBXInCycleRangeLtcyAware = cms.untracked.vint32(19,21)
PotentialTIBTECFrameHeaderEventsFPeak.historyProduct             = cms.untracked.InputTag("ConsecutiveHEs")
PotentialTIBTECFrameHeaderEventsFPeak.APVPhaseLabel              = cms.untracked.string("apvPhases")

PotentialTIBTECFrameHeaderEventsAdditionalPeak = DPGAnalysis.SiStripTools.filters.Potential_TIBTEC_FrameHeaderEvents_firstpeak_cfi.potentialTIBTECFrameHeaderEventsFPeak.clone()
PotentialTIBTECFrameHeaderEventsAdditionalPeak.partitionName              = cms.untracked.string("TI")
PotentialTIBTECFrameHeaderEventsAdditionalPeak.absBXInCycleRangeLtcyAware = cms.untracked.vint32(24,25)
PotentialTIBTECFrameHeaderEventsAdditionalPeak.historyProduct             = cms.untracked.InputTag("ConsecutiveHEs")
PotentialTIBTECFrameHeaderEventsAdditionalPeak.APVPhaseLabel              = cms.untracked.string("apvPhases")

#--------------------------------------------
# Filter to remove high multiplicity events #
#--------------------------------------------

import DPGAnalysis.SiStripTools.largesistripclusterevents_cfi
LargeSiStripClusterEvents = DPGAnalysis.SiStripTools.largesistripclusterevents_cfi.largeSiStripClusterEvents.clone()
LargeSiStripClusterEvents.collectionName = cms.InputTag("calZeroBiasClusters")

#------------
# Sequences #
#------------

seqAPVCycleFilter = cms.Sequence(apvPhases+
                                 ConsecutiveHEs+
                                 ~PotentialTIBTECHugeEvents*
                                 ~PotentialTIBTECFrameHeaderEventsFPeak*
                                 ~PotentialTIBTECFrameHeaderEventsAdditionalPeak)

seqMultiplicityFilter = cms.Sequence(~LargeSiStripClusterEvents)

ALCARECOSiStripCalZeroBiasDQM = cms.Sequence(seqAPVCycleFilter*
                                             seqMultiplicityFilter*
                                             SiStripCalZeroBiasMonitorCluster)
