import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
pixelTracksMonitor = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
pixelTracksMonitor.FolderName                = 'Tracking/PixelTrackParameters/pixelTracks'
pixelTracksMonitor.TrackProducer             = 'pixelTracks'
pixelTracksMonitor.allTrackProducer          = 'pixelTracks'
pixelTracksMonitor.beamSpot                  = 'offlineBeamSpot'
pixelTracksMonitor.primaryVertex             = 'pixelVertices'
pixelTracksMonitor.pvNDOF                    = 1
pixelTracksMonitor.doAllPlots                = False
pixelTracksMonitor.doLumiAnalysis            = True
pixelTracksMonitor.doProfilesVsLS            = True
pixelTracksMonitor.doDCAPlots                = True
pixelTracksMonitor.doProfilesVsLS            = True
pixelTracksMonitor.doPlotsVsGoodPVtx         = True
pixelTracksMonitor.doEffFromHitPatternVsPU   = False
pixelTracksMonitor.doEffFromHitPatternVsBX   = False
pixelTracksMonitor.doEffFromHitPatternVsLUMI = False
pixelTracksMonitor.doPlotsVsGoodPVtx         = True
pixelTracksMonitor.doPlotsVsLUMI             = True
pixelTracksMonitor.doPlotsVsBX               = True

_trackSelector = cms.EDFilter('TrackSelector',
    src = cms.InputTag('pixelTracks'),
    cut = cms.string("")
)

quality = {
    "L"  : "loose",
    "T"  : "tight",
    "HP" : "highPurity",
}

for key,value in quality.items():
    label = "pixelTracks"+key
    print label
    cutstring = "quality('" + value + "')" 
    print cutstring
    locals()[label] = _trackSelector.clone( cut = cutstring )
    locals()[label].setLabel(label)

for key,value in quality.items():
    label = "pixelTracksMonitor"+key
    locals()[label] = pixelTracksMonitor.clone(
        TrackProducer = "pixelTracks"+key,
        FolderName    = "Tracking/PixelTrackParameters/"+value
    )
    locals()[label].setLabel(label)

ntuplet = {
    '3' : "3hits", # ==3
    '4' : "4hits"  # >=4 
}
for kN,vN in ntuplet.items():
    for key,value in quality.items():
        label = "pixelTracks" + vN + key
        print label

        cutstring = "numberOfValidHits == " + kN + " & quality('" + value + "')" 
        print cutstring
        locals()[label] = _trackSelector.clone( cut = cutstring )
        locals()[label].setLabel(label)

for kN,vN in ntuplet.items():
    for key,value in quality.items():
        label = "pixelTracks" + vN + "Monitor" + key
        print label
        locals()[label] = pixelTracksMonitor.clone(
            TrackProducer = "pixelTracks" + vN + key,
            FolderName    = "Tracking/PixelTrackParameters/" + vN + "/" + value
        )
        locals()[label].setLabel(label)


from CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi import goodOfflinePrimaryVertices as _goodOfflinePrimaryVertices
goodPixelVertices = _goodOfflinePrimaryVertices.clone(
    src = "pixelVertices",
)

from DQM.TrackingMonitor.primaryVertexResolution_cfi import primaryVertexResolution as _primaryVertexResolution
pixelVertexResolution = _primaryVertexResolution.clone(
    vertexSrc = "goodPixelVertices",
    rootFolder = "OfflinePixelPV/Resolution",
)

pixelTracksMonitoringTask = cms.Task(
    goodPixelVertices,
)

for category in ["pixelTracks", "pixelTracks3hits", "pixelTracks4hits"]:
    for key in quality:
        label = category+key
        print label
        pixelTracksMonitoringTask.add(locals()[label])

allPixelTracksMonitoring = cms.Sequence()
for category in ["pixelTracksMonitor", "pixelTracks3hitsMonitor", "pixelTracks4hitsMonitor" ]:
    for key in quality:
        label = category+key
        print label
        allPixelTracksMonitoring += locals()[label]

pixelTracksMonitoring = cms.Sequence(
    allPixelTracksMonitoring +
    pixelVertexResolution,
    pixelTracksMonitoringTask
)




