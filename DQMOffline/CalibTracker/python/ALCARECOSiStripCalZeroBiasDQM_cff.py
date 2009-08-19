import FWCore.ParameterSet.Config as cms

#------------------------
# SiStripMonitorCluster #
#------------------------

import DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi

SiStripCalZeroBiasMonitorCluster = DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi.SiStripMonitorCluster.clone()

SiStripCalZeroBiasMonitorCluster.OutputMEsInRootFile                     = False 
SiStripCalZeroBiasMonitorCluster.SelectAllDetectors                      = True 
SiStripCalZeroBiasMonitorCluster.StripQualityLabel                       = 'unbiased'

SiStripCalZeroBiasMonitorCluster.TH1ClusterDigiPos.moduleswitchon        = True
SiStripCalZeroBiasMonitorCluster.TH1ClusterDigiPos.layerswitchon         = False
SiStripCalZeroBiasMonitorCluster.TH1ClusterPos.moduleswitchon            = False
SiStripCalZeroBiasMonitorCluster.TH1ClusterPos.layerswitchon             = False
SiStripCalZeroBiasMonitorCluster.TH1nClusters.moduleswitchon             = False
SiStripCalZeroBiasMonitorCluster.TH1nClusters.layerswitchon              = False
SiStripCalZeroBiasMonitorCluster.TH1ClusterStoN.moduleswitchon           = False
SiStripCalZeroBiasMonitorCluster.TH1ClusterStoN.layerswitchon            = False
SiStripCalZeroBiasMonitorCluster.TH1ClusterStoNVsPos.moduleswitchon      = False
SiStripCalZeroBiasMonitorCluster.TH1ClusterStoNVsPos.layerswitchon       = False
SiStripCalZeroBiasMonitorCluster.TH1ClusterNoise.moduleswitchon          = False
SiStripCalZeroBiasMonitorCluster.TH1ClusterNoise.layerswitchon           = False
SiStripCalZeroBiasMonitorCluster.TH1NrOfClusterizedStrips.moduleswitchon = False
SiStripCalZeroBiasMonitorCluster.TH1NrOfClusterizedStrips.layerswitchon  = False
SiStripCalZeroBiasMonitorCluster.TH1ModuleLocalOccupancy.moduleswitchon  = False
SiStripCalZeroBiasMonitorCluster.TH1ModuleLocalOccupancy.layerswitchon   = False
SiStripCalZeroBiasMonitorCluster.TH1ClusterCharge.moduleswitchon         = False
SiStripCalZeroBiasMonitorCluster.TH1ClusterCharge.layerswitchon          = False
SiStripCalZeroBiasMonitorCluster.TH1ClusterWidth.moduleswitchon          = False
SiStripCalZeroBiasMonitorCluster.TH1ClusterWidth.layerswitchon           = False
SiStripCalZeroBiasMonitorCluster.TProfNumberOfCluster.moduleswitchon     = False
SiStripCalZeroBiasMonitorCluster.TProfNumberOfCluster.layerswitchon      = False
SiStripCalZeroBiasMonitorCluster.TProfClusterWidth.moduleswitchon        = False
SiStripCalZeroBiasMonitorCluster.TProfClusterWidth.layerswitchon         = False
SiStripCalZeroBiasMonitorCluster.TkHistoMap_On                           = False
SiStripCalZeroBiasMonitorCluster.TH1TotalNumberOfClusters.subdetswitchon = True
SiStripCalZeroBiasMonitorCluster.Mod_On                                  = True
SiStripCalZeroBiasMonitorCluster.TH2ClustersApvCycle.subdetswitchon      = True
SiStripCalZeroBiasMonitorCluster.TH2ClustersApvCycle.yfactor             = 0.0002
SiStripCalZeroBiasMonitorCluster.ClusterProducer                         = 'calZeroBiasClusters'

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
