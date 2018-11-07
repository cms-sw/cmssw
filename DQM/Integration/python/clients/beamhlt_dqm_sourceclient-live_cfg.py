from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

#process = cms.Process("BeamMonitor", eras.Run2_2018) # FIMXE
process = cms.Process("BeamMonitor", eras.Run2_2018_pp_on_AA)

# Message logger
#process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring('*'),
#    cerr = cms.untracked.PSet(
#        FwkReport = cms.untracked.PSet(
#            optionalPSet = cms.untracked.bool(True),
#            reportEvery = cms.untracked.int32(1000),
#            limit = cms.untracked.int32(999999)
#        )
#    ),
#    destinations = cms.untracked.vstring('cerr'),
#)

# Common part for PP and H.I Running
#-----------------------------
# for live online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")
# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")

# new stream label
process.source.streamLabel = cms.untracked.string('streamDQMOnlineBeamspot')

#--------------------------
# HLT Filter
# 0=random, 1=physics, 2=calibration, 3=technical
#--------------------------
process.hltTriggerTypeFilter = cms.EDFilter("HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32(1)
)

#-----------------------------
# DQM Live Environment
#-----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'TrackingHLTBeamspotStream'
process.dqmSaver.tag           = 'TrackingHLTBeamspotStream'

#-----------------------------
# BeamMonitor
#-----------------------------
process.load("DQM.BeamMonitor.BeamMonitor_cff")

#---------------
# Calibration
#---------------
# Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Condition for lxplus: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')

# Change Beam Monitor variables
if process.dqmRunConfig.type.value() is "production":
  process.dqmBeamMonitor.BeamFitter.WriteAscii = True
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/dqmpro/BeamMonitorDQM/BeamFitResults.txt'
else:
  process.dqmBeamMonitor.BeamFitter.WriteAscii = False
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/dqmdev/BeamMonitorDQM/BeamFitResults.txt'

process.dqmcommon = cms.Sequence(process.dqmEnv
                               * process.dqmSaver)

process.monitor = cms.Sequence(process.dqmBeamMonitor)

#-----------------------------------------------------------
# process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

#--------------------------
# Proton-Proton Stuff
#--------------------------

if (process.runType.getRunType() == process.runType.pp_run or
    process.runType.getRunType() == process.runType.pp_run_stage1 or
    process.runType.getRunType() == process.runType.cosmic_run or
    process.runType.getRunType() == process.runType.cosmic_run_stage1 or 
    process.runType.getRunType() == process.runType.hpu_run or
    process.runType.getRunType() == process.runType.hi_run):

    print("[beamhlt_dqm_sourceclient-live_cfg]:: Running pp")

    process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

    process.dqmBeamMonitor.monitorName = 'TrackingHLTBeamspotStream'

    process.dqmBeamMonitor.OnlineMode = True              

    process.dqmBeamMonitor.resetEveryNLumi   = 5
    process.dqmBeamMonitor.resetPVEveryNLumi = 5

    process.dqmBeamMonitor.PVFitter.minNrVerticesForFit = 20
    process.dqmBeamMonitor.PVFitter.minVertexNdf        = 10
  
    # some inputs to BeamMonitor
    if(process.runType.getRunType() == process.runType.hi_run):
      process.dqmBeamMonitor.BeamFitter.TrackCollection = 'hltPFMuonMergingPPOnAA'
      process.dqmBeamMonitor.primaryVertex              = 'hltVerticesPFFilterPPOnAA'
      process.dqmBeamMonitor.PVFitter.VertexCollection  = 'hltVerticesPFFilterPPOnAA'
    else:
      process.dqmBeamMonitor.BeamFitter.TrackCollection = 'hltPFMuonMerging'
      process.dqmBeamMonitor.primaryVertex              = 'hltVerticesPFFilter'
      process.dqmBeamMonitor.PVFitter.VertexCollection  = 'hltVerticesPFFilter'

    # keep checking this with new release expected close to 1
    process.dqmBeamMonitor.PVFitter.errorScale = 0.95

    #TriggerName for selecting pv for DIP publication, NO wildcard needed here
    #it will pick all triggers which has these strings in theri name
    process.dqmBeamMonitor.jetTrigger = cms.untracked.vstring(
        "HLT_HT300_Beamspot", "HLT_HT300_Beamspot",
        "HLT_PAZeroBias_v", "HLT_ZeroBias_", "HLT_QuadJet",
        "HLT_HI")

    process.dqmBeamMonitor.hltResults = cms.InputTag("TriggerResults","","HLT")

    process.p = cms.Path( process.hltTriggerTypeFilter
                        * process.dqmcommon
                        * process.offlineBeamSpot
                        * process.monitor )

