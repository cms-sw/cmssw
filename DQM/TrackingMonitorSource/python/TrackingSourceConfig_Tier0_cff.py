import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg
import RecoTracker.IterativeTracking.iterativeTkUtils as _utils

### load which are the tracks collection 2 be monitored
from DQM.TrackingMonitorSource.TrackCollections2monitor_cff import *

### load the different flavour of settings of the TrackingMonitor module
from DQM.TrackingMonitorSource.TrackerCollisionTrackingMonitor_cff import *

#import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
TrackMon_ckf 					   = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
TrackMon_ckf.TKTrackCollection                     = 'generalTracks'#ctfWithMaterialTracksBeamHaloMuon'#rsWithMaterialTracksP5'#muons'#globalCosmicMuons'#ctfWithMaterialTracksP5'
TrackMon_ckf.AlgoName                              = 'CKFTk'
TrackMon_ckf.FolderName                            = 'Tracking/TrackParameters/TrackEfficiency'

# Clone for RS Tracks
#import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
#TrackEffMon_rs = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
#TrackEffMon_rs.TKTrackCollection                   = 'rsWithMaterialTracksP5'
#TrackEffMon_rs.AlgoName                            = 'RSTk'
#TrackEffMon_rs.FolderName                          = 'Tracking/TrackParameters/TrackEfficiency'

# Clone for Beam Halo  Tracks
#import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
#TrackEffMon_bhmuon = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
#TrackEffMon_bhmuon.TKTrackCollection               = 'ctfWithMaterialTracksBeamHaloMuon'
#TrackEffMon_bhmuon.AlgoName                        = 'BHMuonTk'
#TrackEffMon_bhmuon.FolderName                      = 'Tracking/TrackParameters/TrackEfficiency'

# Split Tracking
# from  DQM.TrackingMonitor.TrackSplittingMonitor_cfi import *
# TrackSplitMonitor.FolderName = 'Tracking/TrackParameters/SplitTracks'


# DQM Services
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmInfoTracking = DQMEDAnalyzer('DQMEventInfo',
     subSystemFolder = cms.untracked.string('Tracking')
)


### define one EDAnalyzer per each track collection
### following suggestion 2. in
### https://hypernews.cern.ch/HyperNews/CMS/get/sw-develtools/1908/1.html
for tracks in selectedTracks :
    label = 'TrackerCollisionSelectedTrackMonCommon' + str(tracks)
    locals()[label] = TrackerCollisionTrackMonCommon.clone()
    locals()[label].TrackProducer    = cms.InputTag(tracks)
    locals()[label].allTrackProducer = cms.InputTag(allTrackProducer[tracks])
    locals()[label].FolderName       = cms.string(mainfolderName[tracks])
    locals()[label].PVFolderName     = cms.string(vertexfolderName[tracks])
    locals()[label].TrackPtMin       = trackPtMin[tracks]
    locals()[label].TrackPtBin       = trackPtN[tracks]
    locals()[label].TrackPtMax       = trackPtMax[tracks]
    locals()[label].TrackPBin        = trackPtN[tracks]
    locals()[label].TrackPMin        = trackPtMin[tracks]
    locals()[label].TrackPMax        = trackPtMax[tracks]
    locals()[label].doDCAPlots       = doPlotsPCA[tracks]
    locals()[label].doDCAwrtPVPlots  = doPlotsPCA[tracks]
    locals()[label].doDCAwrt000Plots = doPlotsPCA[tracks]
    locals()[label].doSIPPlots       = doPlotsPCA[tracks]
    locals()[label].numCut           = numCutString[tracks]
    locals()[label].denCut           = denCutString[tracks]
    locals()[label].doGoodTracksPlots                   = doGoodTracksPlots                   [tracks]
    locals()[label].doTrackerSpecific                   = doTrackerSpecific                   [tracks]
    locals()[label].doHitPropertiesPlots                = doHitPropertiesPlots                [tracks]
    locals()[label].doGeneralPropertiesPlots            = doGeneralPropertiesPlots            [tracks]
    locals()[label].doBeamSpotPlots                     = doBeamSpotPlots                     [tracks]
    locals()[label].doSeedParameterHistos               = doSeedParameterHistos               [tracks]
    locals()[label].doRecHitVsPhiVsEtaPerTrack          = doRecHitVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doRecHitVsPtVsEtaPerTrack           = doRecHitVsPtVsEtaPerTrack           [tracks]
    locals()[label].doGoodTrackRecHitVsPhiVsEtaPerTrack = doGoodTrackRecHitVsPhiVsEtaPerTrack [tracks]
    locals()[label].doLayersVsPhiVsEtaPerTrack          = doLayersVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doGoodTrackLayersVsPhiVsEtaPerTrack = doGoodTrackLayersVsPhiVsEtaPerTrack [tracks]
    locals()[label].doPUmonitoring                      = doPUmonitoring                      [tracks]
    locals()[label].doPlotsVsBXlumi                     = doPlotsVsBXlumi                     [tracks]
    locals()[label].doPlotsVsGoodPVtx                   = doPlotsVsGoodPVtx                   [tracks]
    locals()[label].doEffFromHitPatternVsPU             = doEffFromHitPatternVsPU             [tracks]
    locals()[label].doEffFromHitPatternVsLUMI           = doEffFromHitPatternVsLumi           [tracks]
    if tracks == 'generalTracks':
        locals()[label].doEffFromHitPatternVsBX = False
    else:
        locals()[label].doEffFromHitPatternVsBX = doEffFromHitPatternVsBX[tracks]
#    locals()[label].doStopSource                        = doStopSource                        [tracks]    
    locals()[label].setLabel(label)
    
    # ZeroBias
    label = 'TrackerCollisionSelectedTrackMonMB' + str(tracks)                       
    locals()[label] = TrackerCollisionTrackMonMB.clone()
    locals()[label].TrackProducer    = cms.InputTag(tracks)
    locals()[label].FolderName       = cms.string(mainfolderName[tracks])
    locals()[label].PVFolderName     = cms.string(vertexfolderName[tracks])
    locals()[label].TrackPtMin       = trackPtMin[tracks]
    locals()[label].TrackPtBin       = trackPtN[tracks]
    locals()[label].TrackPtMax       = trackPtMax[tracks]
    locals()[label].TrackPBin        = trackPtN[tracks]
    locals()[label].TrackPMin        = trackPtMin[tracks]
    locals()[label].TrackPMax        = trackPtMax[tracks]
    locals()[label].doDCAPlots       = doPlotsPCA[tracks]
    locals()[label].doDCAwrtPVPlots  = doPlotsPCA[tracks]
    locals()[label].doDCAwrt000Plots = doPlotsPCA[tracks]
    locals()[label].doSIPPlots       = doPlotsPCA[tracks]
    locals()[label].numCut           = numCutString[tracks]
    locals()[label].denCut           = denCutString[tracks]
    locals()[label].doGoodTracksPlots                   = doGoodTracksPlots                   [tracks]
    locals()[label].doTrackerSpecific                   = doTrackerSpecific                   [tracks]
    locals()[label].doHitPropertiesPlots                = doHitPropertiesPlots                [tracks]
    locals()[label].doGeneralPropertiesPlots            = doGeneralPropertiesPlots            [tracks]
    locals()[label].doBeamSpotPlots                     = doBeamSpotPlots                     [tracks]
    locals()[label].doSeedParameterHistos               = doSeedParameterHistos               [tracks]
    locals()[label].doRecHitVsPhiVsEtaPerTrack          = doRecHitVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doRecHitVsPtVsEtaPerTrack           = doRecHitVsPtVsEtaPerTrack           [tracks]
    locals()[label].doGoodTrackRecHitVsPhiVsEtaPerTrack = doGoodTrackRecHitVsPhiVsEtaPerTrack [tracks]
    locals()[label].doLayersVsPhiVsEtaPerTrack          = doLayersVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doGoodTrackLayersVsPhiVsEtaPerTrack = doGoodTrackLayersVsPhiVsEtaPerTrack [tracks]
    locals()[label].doPUmonitoring                      = doPUmonitoring                      [tracks]
    locals()[label].doPlotsVsBXlumi                     = doPlotsVsBXlumi                     [tracks]
    locals()[label].doPlotsVsGoodPVtx                   = doPlotsVsGoodPVtx                   [tracks]
    locals()[label].doPlotsVsLUMI                       = cms.bool(True)
    locals()[label].doPlotsVsBX                         = cms.bool(True)
    locals()[label].doEffFromHitPatternVsPU             = doEffFromHitPatternVsPU             [tracks]
    locals()[label].doEffFromHitPatternVsBX             = doEffFromHitPatternVsBX             [tracks]
    locals()[label].doEffFromHitPatternVsLUMI           = cms.bool(True)
    locals()[label].doStopSource                        = doStopSource                        [tracks]    
    locals()[label].setLabel(label)

    # ZeroBias, no hip, no OOT pu (1st collision after abort gap) ###
    label = 'TrackerCollisionSelectedTrackMonZBnoHIPnoOOT' + str(tracks)                       
    locals()[label] = TrackerCollisionTrackMonZBnoHIPnoOOT.clone()
    locals()[label].TrackProducer    = cms.InputTag(tracks)
    locals()[label].FolderName       = cms.string(mainfolderName[tracks]+"/noHIP_noOOT_INpu")
    locals()[label].PVFolderName     = cms.string(vertexfolderName[tracks]+"/noHIP_noOOT_INpu")
    locals()[label].TrackPtMin       = trackPtMin[tracks]
    locals()[label].TrackPtBin       = trackPtN[tracks]
    locals()[label].TrackPtMax       = trackPtMax[tracks]
    locals()[label].TrackPBin        = trackPtN[tracks]
    locals()[label].TrackPMin        = trackPtMin[tracks]
    locals()[label].TrackPMax        = trackPtMax[tracks]
    locals()[label].doDCAPlots       = doPlotsPCA[tracks]
    locals()[label].doDCAwrtPVPlots  = doPlotsPCA[tracks]
    locals()[label].doDCAwrt000Plots = doPlotsPCA[tracks]
    locals()[label].doSIPPlots       = doPlotsPCA[tracks]
    locals()[label].numCut           = numCutString[tracks]
    locals()[label].denCut           = denCutString[tracks]
    locals()[label].doGoodTracksPlots                   = doGoodTracksPlots                   [tracks]
    locals()[label].doTrackerSpecific                   = doTrackerSpecific                   [tracks]
    locals()[label].doHitPropertiesPlots                = doHitPropertiesPlots                [tracks]
    locals()[label].doGeneralPropertiesPlots            = doGeneralPropertiesPlots            [tracks]
    locals()[label].doBeamSpotPlots                     = doBeamSpotPlots                     [tracks]
    locals()[label].doSeedParameterHistos               = doSeedParameterHistos               [tracks]
    locals()[label].doRecHitVsPhiVsEtaPerTrack          = doRecHitVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doRecHitVsPtVsEtaPerTrack           = doRecHitVsPtVsEtaPerTrack           [tracks]
    locals()[label].doGoodTrackRecHitVsPhiVsEtaPerTrack = doGoodTrackRecHitVsPhiVsEtaPerTrack [tracks]
    locals()[label].doLayersVsPhiVsEtaPerTrack          = doLayersVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doGoodTrackLayersVsPhiVsEtaPerTrack = doGoodTrackLayersVsPhiVsEtaPerTrack [tracks]
    locals()[label].doPUmonitoring                      = doPUmonitoring                      [tracks]
    locals()[label].doPlotsVsBXlumi                     = doPlotsVsBXlumi                     [tracks]
    locals()[label].doPlotsVsGoodPVtx                   = doPlotsVsGoodPVtx                   [tracks]
    locals()[label].doPlotsVsLUMI                       = cms.bool(True)
    locals()[label].doPlotsVsBX                         = cms.bool(True)
    locals()[label].doEffFromHitPatternVsPU             = doEffFromHitPatternVsPU             [tracks]
    locals()[label].doEffFromHitPatternVsBX             = doEffFromHitPatternVsBX             [tracks]
    locals()[label].doEffFromHitPatternVsLUMI           = False
    locals()[label].doStopSource                        = doStopSource                        [tracks]    
    locals()[label].setLabel(label)

    # ZeroBias, hip, no OOT pu (1st collision in train) ###
    label = 'TrackerCollisionSelectedTrackMonZBHIPnoOOT' + str(tracks)                       
    locals()[label] = TrackerCollisionTrackMonZBHIPnoOOT.clone()
    locals()[label].TrackProducer    = cms.InputTag(tracks)
    locals()[label].FolderName       = cms.string(mainfolderName[tracks]+"/HIP_noOOT_INpu")
    locals()[label].PVFolderName     = cms.string(vertexfolderName[tracks]+"/HIP_noOOT_INpu")
    locals()[label].TrackPtMin       = trackPtMin[tracks]
    locals()[label].TrackPtBin       = trackPtN[tracks]
    locals()[label].TrackPtMax       = trackPtMax[tracks]
    locals()[label].TrackPBin        = trackPtN[tracks]
    locals()[label].TrackPMin        = trackPtMin[tracks]
    locals()[label].TrackPMax        = trackPtMax[tracks]
    locals()[label].doDCAPlots       = doPlotsPCA[tracks]
    locals()[label].doDCAwrtPVPlots  = doPlotsPCA[tracks]
    locals()[label].doDCAwrt000Plots = doPlotsPCA[tracks]
    locals()[label].doSIPPlots       = doPlotsPCA[tracks]
    locals()[label].numCut           = numCutString[tracks]
    locals()[label].denCut           = denCutString[tracks]
    locals()[label].doGoodTracksPlots                   = doGoodTracksPlots                   [tracks]
    locals()[label].doTrackerSpecific                   = doTrackerSpecific                   [tracks]
    locals()[label].doHitPropertiesPlots                = doHitPropertiesPlots                [tracks]
    locals()[label].doGeneralPropertiesPlots            = doGeneralPropertiesPlots            [tracks]
    locals()[label].doBeamSpotPlots                     = doBeamSpotPlots                     [tracks]
    locals()[label].doSeedParameterHistos               = doSeedParameterHistos               [tracks]
    locals()[label].doRecHitVsPhiVsEtaPerTrack          = doRecHitVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doRecHitVsPtVsEtaPerTrack           = doRecHitVsPtVsEtaPerTrack           [tracks]
    locals()[label].doGoodTrackRecHitVsPhiVsEtaPerTrack = doGoodTrackRecHitVsPhiVsEtaPerTrack [tracks]
    locals()[label].doLayersVsPhiVsEtaPerTrack          = doLayersVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doGoodTrackLayersVsPhiVsEtaPerTrack = doGoodTrackLayersVsPhiVsEtaPerTrack [tracks]
    locals()[label].doPUmonitoring                      = doPUmonitoring                      [tracks]
    locals()[label].doPlotsVsBXlumi                     = doPlotsVsBXlumi                     [tracks]
    locals()[label].doPlotsVsGoodPVtx                   = doPlotsVsGoodPVtx                   [tracks]
    locals()[label].doPlotsVsLUMI                       = cms.bool(True)
    locals()[label].doPlotsVsBX                         = cms.bool(True)
    locals()[label].doEffFromHitPatternVsPU             = doEffFromHitPatternVsPU             [tracks]
    locals()[label].doEffFromHitPatternVsBX             = doEffFromHitPatternVsBX             [tracks]
    locals()[label].doEffFromHitPatternVsLUMI           = False
    locals()[label].doStopSource                        = doStopSource                        [tracks]    
    locals()[label].setLabel(label)


    # ZeroBias, hip, OOT pu (1st collision after train) ###
    label = 'TrackerCollisionSelectedTrackMonZBHIPOOT' + str(tracks)                       
    locals()[label] = TrackerCollisionTrackMonZBHIPOOT.clone()
    locals()[label].TrackProducer    = cms.InputTag(tracks)
    locals()[label].FolderName       = cms.string(mainfolderName[tracks]+"/HIP_OOT_noINpu")
    locals()[label].PVFolderName     = cms.string(vertexfolderName[tracks]+"/HIP_OOT_noINpu")
    locals()[label].TrackPtMin       = trackPtMin[tracks]
    locals()[label].TrackPtBin       = trackPtN[tracks]
    locals()[label].TrackPtMax       = trackPtMax[tracks]
    locals()[label].TrackPBin        = trackPtN[tracks]
    locals()[label].TrackPMin        = trackPtMin[tracks]
    locals()[label].TrackPMax        = trackPtMax[tracks]
    locals()[label].doDCAPlots       = doPlotsPCA[tracks]
    locals()[label].doDCAwrtPVPlots  = doPlotsPCA[tracks]
    locals()[label].doDCAwrt000Plots = doPlotsPCA[tracks]
    locals()[label].doSIPPlots       = doPlotsPCA[tracks]
    locals()[label].numCut           = numCutString[tracks]
    locals()[label].denCut           = denCutString[tracks]
    locals()[label].doGoodTracksPlots                   = doGoodTracksPlots                   [tracks]
    locals()[label].doTrackerSpecific                   = doTrackerSpecific                   [tracks]
    locals()[label].doHitPropertiesPlots                = doHitPropertiesPlots                [tracks]
    locals()[label].doGeneralPropertiesPlots            = doGeneralPropertiesPlots            [tracks]
    locals()[label].doBeamSpotPlots                     = doBeamSpotPlots                     [tracks]
    locals()[label].doSeedParameterHistos               = doSeedParameterHistos               [tracks]
    locals()[label].doRecHitVsPhiVsEtaPerTrack          = doRecHitVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doRecHitVsPtVsEtaPerTrack           = doRecHitVsPtVsEtaPerTrack           [tracks]
    locals()[label].doGoodTrackRecHitVsPhiVsEtaPerTrack = doGoodTrackRecHitVsPhiVsEtaPerTrack [tracks]
    locals()[label].doLayersVsPhiVsEtaPerTrack          = doLayersVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doGoodTrackLayersVsPhiVsEtaPerTrack = doGoodTrackLayersVsPhiVsEtaPerTrack [tracks]
    locals()[label].doPUmonitoring                      = doPUmonitoring                      [tracks]
    locals()[label].doPlotsVsBXlumi                     = doPlotsVsBXlumi                     [tracks]
    locals()[label].doPlotsVsGoodPVtx                   = doPlotsVsGoodPVtx                   [tracks]
    locals()[label].doPlotsVsLUMI                       = cms.bool(True)
    locals()[label].doPlotsVsBX                         = cms.bool(True)
    locals()[label].doEffFromHitPatternVsPU             = doEffFromHitPatternVsPU             [tracks]
    locals()[label].doEffFromHitPatternVsBX             = doEffFromHitPatternVsBX             [tracks]
    locals()[label].doEffFromHitPatternVsLUMI           = False
    locals()[label].doStopSource                        = doStopSource                        [tracks]    
    locals()[label].setLabel(label)


#-------------------------------------------------
# Tracking Monitor 
#-------------------------------------------------
import DQM.TrackingMonitor.TrackingMonitorSeed_cfi

from DQM.TrackingMonitorSource.IterTrackingModules4seedMonitoring_cfi import *
# Create first modules for all possible iterations, select later which
# ones to actually use based on era
def _copyIfExists(mod, pset, name):
    if hasattr(pset, name):
        setattr(mod, name, getattr(pset, name))
for _step, _pset in seedMonitoring.items():
    _mod = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone(
        doTrackCandHistos = cms.bool(True)
    )
    locals()['TrackSeedMon'+str(_step)] = _mod
    _mod.TrackProducer = cms.InputTag("generalTracks")
    _mod.FolderName = cms.string("Tracking/TrackParameters/generalTracks/SeedMon/"+str(_step))
    _mod.doPUmonitoring = cms.bool(False)
    _mod.doLumiAnalysis = cms.bool(False)
    _mod.doPlotsVsGoodPVtx = cms.bool(False)
    _mod.SeedProducer  = _pset.seedInputTag
    _mod.TCProducer    = _pset.trackCandInputTag
    _mod.AlgoName      = cms.string( str(_step) )
    _mod.TkSeedSizeBin = _pset.trackSeedSizeBin
    _mod.TkSeedSizeMin = _pset.trackSeedSizeMin
    _mod.TkSeedSizeMax = _pset.trackSeedSizeMax
    _mod.ClusterLabels = _pset.clusterLabel
    if _pset.clusterLabel == cms.vstring('Pix') :
        _mod.NClusPxBin = _pset.clusterBin
        _mod.NClusPxMax = _pset.clusterMax
    elif _pset.clusterLabel == cms.vstring('Strip') or _pset.clusterLabel == cms.vstring('Tot') :
        _mod.NClusStrBin = _pset.clusterBin
        _mod.NClusStrMax = _pset.clusterMax
    if hasattr(_pset, "RegionProducer") or hasattr(_pset, "RegionSeedingLayersProducer"):
        _mod.doRegionPlots = True
        _copyIfExists(_mod, _pset, "RegionProducer")
        _copyIfExists(_mod, _pset, "RegionSeedingLayersProducer")
        _copyIfExists(_mod, _pset, "RegionSizeBin")
        _copyIfExists(_mod, _pset, "RegionSizeMax")
        if hasattr(_pset, "RegionCandidates"):
            _mod.doRegionCandidatePlots = True
            _mod.RegionCandidates = _pset.RegionCandidates
    if hasattr(_pset, "trajCandPerSeedBin"):
        _mod.SeedCandBin = _pset.trajCandPerSeedBin
        _mod.SeedCandMax = _pset.trajCandPerSeedMax

# DQM Services
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmInfoTracking = DQMEDAnalyzer('DQMEventInfo',
    subSystemFolder = cms.untracked.string('Tracking')
)

# LogMessageMonitor ####
### load which are the module to monitor
from DQM.TrackingMonitorSource.EDModules2monitor_cfi import *

### load the different flavour of settings of the LogMessageMonitor module
from DQM.TrackingMonitorSource.LogMessageMonitor_cff import *

for module in selectedModules :
    label = str(module)+'LogMessageMonCommon'
    locals()[label] = LogMessageMonCommon.clone()
    locals()[label].pluginsMonName = pluginsMonName[module]
    locals()[label].modules        = modulesLabel[module]
    locals()[label].categories     = categories[module]
    locals()[label].setLabel(label)

    label = str(module)+'LogMessageMonMB'
    locals()[label] = LogMessageMonMB.clone()
    locals()[label].pluginsMonName = pluginsMonName[module]
    locals()[label].modules        = modulesLabel[module]
    locals()[label].categories     = categories[module]
    locals()[label].setLabel(label)


# dEdx monitor ####
### load which dedx
from DQM.TrackingMonitorSource.dedxHarmonic2monitor_cfi import *

### load the different flavour of settings of the dEdxAnalyzer module
from DQM.TrackingMonitorSource.dEdxAnalyzer_cff import *


# temporary patch in order to have BXlumi 
from RecoLuminosity.LumiProducer.lumiProducer_cff import *

# import v0 monitoring
from DQM.TrackingMonitor.V0Monitor_cff import *

# better clone for now because goodOfflinePrimaryVertices is used also
# within the reco sequence, and without cloning framework will throw
# "unrunnable schedule" exception for workflows without --runUnscheduled
from CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi import goodOfflinePrimaryVertices
trackingDQMgoodOfflinePrimaryVertices = goodOfflinePrimaryVertices.clone()


# import PV resolution
from DQM.TrackingMonitor.primaryVertexResolution_cfi import *
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(primaryVertexResolution, forceSCAL = False)
# Sequence
TrackingDQMSourceTier0 = cms.Sequence(cms.ignore(trackingDQMgoodOfflinePrimaryVertices))
# dEdx monitoring
TrackingDQMSourceTier0 += dedxHarmonicSequence * dEdxMonCommon * dEdxHitMonCommon * TrackMon_ckf * dqmInfoTracking
#TrackMon_cosmicTk*TrackMon_ckf*TrackEffMon_ckf*TrackSplitMonitor*dqmInfoTracking
#    # temporary patch in order to have BXlumi
#    * lumiProducer
# track collections
for tracks in selectedTracks :
    if tracks != 'generalTracks':
        TrackingDQMSourceTier0 += cms.ignore(sequenceName[tracks])
    label = 'TrackerCollisionSelectedTrackMonCommon' + str(tracks)
    TrackingDQMSourceTier0 += cms.ignore(locals()[label])
# seeding monitoring
for _eraName, _postfix, _era in _cfg.allEras():
    mvaSel = _utils.getMVASelectors(_postfix)
    _seq = cms.Sequence()
    for step in locals()["selectedIterTrackingStep"+_postfix]:
        if step in mvaSel:
            locals()["TrackSeedMon"+step].doMVAPlots = True
            locals()["TrackSeedMon"+step].TrackProducerForMVA = mvaSel[step][0]
            locals()["TrackSeedMon"+step].MVAProducers = mvaSel[step][1]

        _seq += locals()["TrackSeedMon"+step]
    if _eraName == "":
        locals()["TrackSeedMonSequence"] = _seq
    else:
        _era.toReplaceWith(TrackSeedMonSequence, _seq)

_seedingDeepCore_TrackSeedMonSequence = TrackSeedMonSequence.copy()
_seedingDeepCore_TrackSeedMonSequence.remove(locals()["TrackSeedMonjetCoreRegionalStep"])
#_seedingDeepCore_TrackSeedMonSequence += (locals()["TrackSeedMonjetCoreRegionalStepBarrel"])
_seedingDeepCore_TrackSeedMonSequence += (locals()["TrackSeedMonjetCoreRegionalStepEndcap"])
from Configuration.ProcessModifiers.seedingDeepCore_cff import seedingDeepCore
seedingDeepCore.toReplaceWith(TrackSeedMonSequence,_seedingDeepCore_TrackSeedMonSequence)

TrackingDQMSourceTier0 += TrackSeedMonSequence

# MessageLog
for module in selectedModules :
    label = str(module)+'LogMessageMonCommon'
    TrackingDQMSourceTier0 += locals()[label]
TrackingDQMSourceTier0 += voMonitoringSequence
TrackingDQMSourceTier0 += voWcutMonitoringSequence
TrackingDQMSourceTier0 += primaryVertexResolution
TrackingDQMSourceTier0 += dqmInfoTracking


TrackingDQMSourceTier0Common = cms.Sequence(cms.ignore(trackingDQMgoodOfflinePrimaryVertices))
# dEdx monitoring
TrackingDQMSourceTier0Common += (dedxHarmonicSequence * dEdxMonCommon * dEdxHitMonCommon * TrackMon_ckf * dqmInfoTracking)
#TrackEffMon_ckf*TrackSplitMonitor*dqmInfoTracking)    
## monitor track collections
for tracks in selectedTracks :
    if tracks != 'generalTracks':
        TrackingDQMSourceTier0Common+=cms.ignore(sequenceName[tracks])
    label = 'TrackerCollisionSelectedTrackMonCommon' + str(tracks)
    TrackingDQMSourceTier0Common += cms.ignore(locals()[label])
# seeding monitoring
TrackingDQMSourceTier0Common += TrackSeedMonSequence
# MessageLog
for module in selectedModules :
    label = str(module)+'LogMessageMonCommon'
    TrackingDQMSourceTier0Common += locals()[label]
TrackingDQMSourceTier0Common += voMonitoringCommonSequence
TrackingDQMSourceTier0Common += voWcutMonitoringCommonSequence
TrackingDQMSourceTier0Common += primaryVertexResolution
TrackingDQMSourceTier0Common += dqmInfoTracking

TrackingDQMSourceTier0MinBias = cms.Sequence(cms.ignore(trackingDQMgoodOfflinePrimaryVertices))
# dEdx monitoring
TrackingDQMSourceTier0MinBias += dedxHarmonicSequence * dEdxMonCommon * dEdxHitMonCommon * TrackMon_ckf * dqmInfoTracking
#TrackMon_cosmicTk*TrackMon_ckf*TrackEffMon_ckf*TrackSplitMonitor*dqmInfoTracking#TrackMon_ckf*TrackEffMon_ckf 
#    * lumiProducer
# monitor track collections
for tracks in selectedTracks :
    if tracks != 'generalTracks':
        TrackingDQMSourceTier0MinBias += cms.ignore(sequenceName[tracks])

    for topology in [ 'MB', 'ZBnoHIPnoOOT', 'ZBHIPnoOOT', 'ZBHIPOOT']:
        label = 'TrackerCollisionSelectedTrackMon' + str(topology) + str(tracks)
        TrackingDQMSourceTier0MinBias += cms.ignore(locals()[label])
# seeding monitoring
TrackingDQMSourceTier0MinBias += TrackSeedMonSequence
# MessageLog
for module in selectedModules :
    label = str(module)+'LogMessageMonMB'
    TrackingDQMSourceTier0MinBias += locals()[label]
# V0 monitoring
TrackingDQMSourceTier0MinBias += voMonitoringMBSequence
TrackingDQMSourceTier0MinBias += voMonitoringZBnoHIPnoOOTSequence
TrackingDQMSourceTier0MinBias += voMonitoringZBHIPnoOOTSequence
TrackingDQMSourceTier0MinBias += voMonitoringZBHIPOOTSequence
TrackingDQMSourceTier0MinBias += voWcutMonitoringMBSequence
TrackingDQMSourceTier0MinBias += voWcutMonitoringZBnoHIPnoOOTSequence
TrackingDQMSourceTier0MinBias += voWcutMonitoringZBHIPnoOOTSequence
TrackingDQMSourceTier0MinBias += voWcutMonitoringZBHIPOOTSequence
# PV resolution
TrackingDQMSourceTier0MinBias += primaryVertexResolution

TrackingDQMSourceTier0MinBias += dqmInfoTracking

