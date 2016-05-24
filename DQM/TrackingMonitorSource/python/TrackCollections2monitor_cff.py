import FWCore.ParameterSet.Config as cms

mainfolderName   = {}
vertexfolderName = {}
sequenceName = {}
trackPtN   = {}
trackPtMin = {}
trackPtMax = {}
doPlotsPCA = {}
numCutString = {}
denCutString = {}
doGoodTracksPlots                   = {}
doTrackerSpecific                   = {}
doHitPropertiesPlots                = {}
doGeneralPropertiesPlots            = {}
doBeamSpotPlots                     = {}
doSeedParameterHistos               = {}
doRecHitVsPhiVsEtaPerTrack          = {}
doGoodTrackRecHitVsPhiVsEtaPerTrack = {}
doLayersVsPhiVsEtaPerTrack          = {}
doGoodTrackLayersVsPhiVsEtaPerTrack = {}
doPUmonitoring                      = {}
doPlotsVsBXlumi                     = {}
doPlotsVsGoodPVtx                   = {}
doEffFromHitPatternVsPU             = {}
doEffFromHitPatternVsBX             = {}
doStopSource                        = {}

selectedTracks = []

mainfolderName  ['generalTracks'] = 'Tracking/TrackParameters/generalTracks'
vertexfolderName['generalTracks'] = 'Tracking/PrimaryVertices/generalTracks'
trackPtN        ['generalTracks'] = cms.int32(100)
trackPtMin      ['generalTracks'] = cms.double(0.)
trackPtMax      ['generalTracks'] = cms.double(100.)
doPlotsPCA      ['generalTracks'] = cms.bool(False)
numCutString    ['generalTracks'] = cms.string("")
denCutString    ['generalTracks'] = cms.string("")
doGoodTracksPlots                   ['generalTracks'] = cms.bool(True)
doTrackerSpecific                   ['generalTracks'] = cms.bool(True)
doHitPropertiesPlots                ['generalTracks'] = cms.bool(True)
doGeneralPropertiesPlots            ['generalTracks'] = cms.bool(True)
doBeamSpotPlots                     ['generalTracks'] = cms.bool(True)
doSeedParameterHistos               ['generalTracks'] = cms.bool(False)
doRecHitVsPhiVsEtaPerTrack          ['generalTracks'] = cms.bool(True)
doGoodTrackRecHitVsPhiVsEtaPerTrack ['generalTracks'] = cms.bool(True)
doLayersVsPhiVsEtaPerTrack          ['generalTracks'] = cms.bool(True)
doGoodTrackLayersVsPhiVsEtaPerTrack ['generalTracks'] = cms.bool(True)
doPUmonitoring                      ['generalTracks'] = cms.bool(False)
doPlotsVsBXlumi                     ['generalTracks'] = cms.bool(False)
doPlotsVsGoodPVtx                   ['generalTracks'] = cms.bool(True)
doEffFromHitPatternVsPU             ['generalTracks'] = cms.bool(True)
doEffFromHitPatternVsBX             ['generalTracks'] = cms.bool(True)
doStopSource                        ['generalTracks'] = cms.bool(True)

trackSelector = cms.EDFilter('TrackSelector',
    src = cms.InputTag('generalTracks'),
    cut = cms.string("")
)

### highpurity definition: https://cmssdt.cern.ch/SDT/lxr/source/RecoTracker/FinalTrackSelectors/python/selectHighPurity_cfi.py
highPurityPtRange0to1 = trackSelector.clone()
highPurityPtRange0to1.cut = cms.string("quality('highPurity') & pt >= 0 & pt < 1 ")

sequenceName    ['highPurityPtRange0to1'] = highPurityPtRange0to1
mainfolderName  ['highPurityPtRange0to1'] = 'Tracking/TrackParameters/highPurityTracks/pt_0to1'
vertexfolderName['highPurityPtRange0to1'] = 'Tracking/PrimaryVertices/highPurityTracks/pt_0to1'
trackPtN        ['highPurityPtRange0to1'] = cms.int32(10)
trackPtMin      ['highPurityPtRange0to1'] = cms.double(0.)
trackPtMax      ['highPurityPtRange0to1'] = cms.double(1.)
numCutString    ['highPurityPtRange0to1'] = cms.string("") # default: " pt >= 1 & quality('highPurity') "
denCutString    ['highPurityPtRange0to1'] = cms.string(" pt >= 0 & pt < 1 ") # it is as in the default config (just be sure)
doPlotsPCA      ['highPurityPtRange0to1'] = cms.bool(False)
doGoodTracksPlots                   ['highPurityPtRange0to1'] = cms.bool(False)
doTrackerSpecific                   ['highPurityPtRange0to1'] = cms.bool(False)
doHitPropertiesPlots                ['highPurityPtRange0to1'] = cms.bool(True)
doGeneralPropertiesPlots            ['highPurityPtRange0to1'] = cms.bool(True)
doBeamSpotPlots                     ['highPurityPtRange0to1'] = cms.bool(True)
doSeedParameterHistos               ['highPurityPtRange0to1'] = cms.bool(False)
doRecHitVsPhiVsEtaPerTrack          ['highPurityPtRange0to1'] = cms.bool(True)
doGoodTrackRecHitVsPhiVsEtaPerTrack ['highPurityPtRange0to1'] = cms.bool(False)
doLayersVsPhiVsEtaPerTrack          ['highPurityPtRange0to1'] = cms.bool(True)
doGoodTrackLayersVsPhiVsEtaPerTrack ['highPurityPtRange0to1'] = cms.bool(False)
doPUmonitoring                      ['highPurityPtRange0to1'] = cms.bool(True)
doPlotsVsBXlumi                     ['highPurityPtRange0to1'] = cms.bool(False)
doPlotsVsGoodPVtx                   ['highPurityPtRange0to1'] = cms.bool(True)
doEffFromHitPatternVsPU             ['highPurityPtRange0to1'] = cms.bool(False)
doEffFromHitPatternVsBX             ['highPurityPtRange0to1'] = cms.bool(False)
doStopSource                        ['highPurityPtRange0to1'] = cms.bool(True)

highPurityPtRange1to10 = trackSelector.clone()
highPurityPtRange1to10.cut = cms.string("quality('highPurity') & pt >= 1 & pt < 10 ")

sequenceName    ['highPurityPtRange1to10'] = highPurityPtRange1to10 
mainfolderName  ['highPurityPtRange1to10'] = 'Tracking/TrackParameters/highPurityTracks/pt_1to10'
vertexfolderName['highPurityPtRange1to10'] = 'Tracking/PrimaryVertices/highPurityTracks/pt_1to10'
trackPtN        ['highPurityPtRange1to10'] = cms.int32(10)
trackPtMin      ['highPurityPtRange1to10'] = cms.double(1.)
trackPtMax      ['highPurityPtRange1to10'] = cms.double(10.)
numCutString    ['highPurityPtRange1to10'] = cms.string("") # default: " pt >= 1 & quality('highPurity') "
denCutString    ['highPurityPtRange1to10'] = cms.string(" pt >= 1 & pt < 10 ") # it is as in the default config (just be sure)
doGoodTracksPlots                   ['highPurityPtRange1to10'] = cms.bool(True)
doTrackerSpecific                   ['highPurityPtRange1to10'] = cms.bool(True)
doHitPropertiesPlots                ['highPurityPtRange1to10'] = cms.bool(True)
doGeneralPropertiesPlots            ['highPurityPtRange1to10'] = cms.bool(True)
doBeamSpotPlots                     ['highPurityPtRange1to10'] = cms.bool(True)
doSeedParameterHistos               ['highPurityPtRange1to10'] = cms.bool(False)
doRecHitVsPhiVsEtaPerTrack          ['highPurityPtRange1to10'] = cms.bool(True)
doGoodTrackRecHitVsPhiVsEtaPerTrack ['highPurityPtRange1to10'] = cms.bool(True)
doLayersVsPhiVsEtaPerTrack          ['highPurityPtRange1to10'] = cms.bool(True)
doGoodTrackLayersVsPhiVsEtaPerTrack ['highPurityPtRange1to10'] = cms.bool(True)
doPUmonitoring                      ['highPurityPtRange1to10'] = cms.bool(False)
doPlotsVsBXlumi                     ['highPurityPtRange1to10'] = cms.bool(False)
doPlotsVsGoodPVtx                   ['highPurityPtRange1to10'] = cms.bool(True)
doEffFromHitPatternVsPU             ['highPurityPtRange1to10'] = cms.bool(True)
doEffFromHitPatternVsBX             ['highPurityPtRange1to10'] = cms.bool(True)
doStopSource                        ['highPurityPtRange1to10'] = cms.bool(True)

highPurityPt10 = trackSelector.clone()
highPurityPt10.cut = cms.string("quality('highPurity') & pt >= 10")

sequenceName    ['highPurityPt10'] = highPurityPt10 
mainfolderName  ['highPurityPt10'] = 'Tracking/TrackParameters/highPurityTracks/pt_10'
vertexfolderName['highPurityPt10'] = 'Tracking/PrimaryVertices/highPurityTracks/pt_10'
trackPtN        ['highPurityPt10'] = cms.int32(100)
trackPtMin      ['highPurityPt10'] = cms.double(10.)
trackPtMax      ['highPurityPt10'] = cms.double(110.)
numCutString    ['highPurityPt10'] = cms.string("") # default: " pt >= 1 & quality('highPurity') "
denCutString    ['highPurityPt10'] = cms.string(" pt >= 10 ") # it is as in the default config (just be sure)
doGoodTracksPlots                   ['highPurityPt10'] = cms.bool(True)
doTrackerSpecific                   ['highPurityPt10'] = cms.bool(True)
doHitPropertiesPlots                ['highPurityPt10'] = cms.bool(True)
doGeneralPropertiesPlots            ['highPurityPt10'] = cms.bool(True)
doBeamSpotPlots                     ['highPurityPt10'] = cms.bool(True)
doSeedParameterHistos               ['highPurityPt10'] = cms.bool(False)
doRecHitVsPhiVsEtaPerTrack          ['highPurityPt10'] = cms.bool(True)
doGoodTrackRecHitVsPhiVsEtaPerTrack ['highPurityPt10'] = cms.bool(True)
doLayersVsPhiVsEtaPerTrack          ['highPurityPt10'] = cms.bool(True)
doGoodTrackLayersVsPhiVsEtaPerTrack ['highPurityPt10'] = cms.bool(True)
doPUmonitoring                      ['highPurityPt10'] = cms.bool(False)
doPlotsVsBXlumi                     ['highPurityPt10'] = cms.bool(False)
doPlotsVsGoodPVtx                   ['highPurityPt10'] = cms.bool(True)
doEffFromHitPatternVsPU             ['highPurityPt10'] = cms.bool(True)
doEffFromHitPatternVsBX             ['highPurityPt10'] = cms.bool(True)
doStopSource                        ['highPurityPt10'] = cms.bool(True)

###### old monitored track collections
highPurityPt1 = trackSelector.clone()
highPurityPt1.cut = cms.string("quality('highPurity') & pt >= 1")

sequenceName    ['highPurityPt1'] = highPurityPt1
mainfolderName  ['highPurityPt1'] = 'Tracking/TrackParameters/highPurityTracks/pt_1'
vertexfolderName['highPurityPt1'] = 'Tracking/PrimaryVertices/highPurityTracks/pt_1'
trackPtN        ['highPurityPt1'] = cms.int32(100)
trackPtMin      ['highPurityPt1'] = cms.double(0.)
trackPtMax      ['highPurityPt1'] = cms.double(100.)
doPlotsPCA      ['highPurityPt1'] = cms.bool(True)
numCutString    ['highPurityPt1'] = cms.string("") # default: " pt >= 1 & quality('highPurity') "
denCutString    ['highPurityPt1'] = cms.string(" pt >= 1 ") # it is as in the default config (just be sure)
doGoodTracksPlots                   ['highPurityPt1'] = cms.bool(True)
doTrackerSpecific                   ['highPurityPt1'] = cms.bool(True)
doHitPropertiesPlots                ['highPurityPt1'] = cms.bool(True)
doGeneralPropertiesPlots            ['highPurityPt1'] = cms.bool(True)
doBeamSpotPlots                     ['highPurityPt1'] = cms.bool(True)
doSeedParameterHistos               ['highPurityPt1'] = cms.bool(False)
doRecHitVsPhiVsEtaPerTrack          ['highPurityPt1'] = cms.bool(True)
doGoodTrackRecHitVsPhiVsEtaPerTrack ['highPurityPt1'] = cms.bool(True)
doLayersVsPhiVsEtaPerTrack          ['highPurityPt1'] = cms.bool(True)
doGoodTrackLayersVsPhiVsEtaPerTrack ['highPurityPt1'] = cms.bool(True)
doPUmonitoring                      ['highPurityPt1'] = cms.bool(False)
doPlotsVsBXlumi                     ['highPurityPt1'] = cms.bool(False)
doPlotsVsGoodPVtx                   ['highPurityPt1'] = cms.bool(True)
doEffFromHitPatternVsPU             ['highPurityPt1'] = cms.bool(True)
doEffFromHitPatternVsBX             ['highPurityPt1'] = cms.bool(True)
doStopSource                        ['highPurityPt1'] = cms.bool(True)

selectedTracks.extend( ['generalTracks'] )
#selectedTracks.extend( ['highPurityPtRange0to1']  )
#selectedTracks.extend( ['highPurityPtRange1to10'] )
#selectedTracks.extend( ['highPurityPt10']         )

selectedTracks.extend( ['highPurityPt1'] )
selectedTracks.extend( ['highPurityPtRange0to1'] )

#selectedTracks2runSequence=cms.Sequence()
#for tracks in selectedTracks :
#    if tracks != 'generalTracks':
#        selectedTracks2runSequence+=sequenceName[tracks]

