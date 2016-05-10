import FWCore.ParameterSet.Config as cms

mainfolderName   = {}
vertexfolderName = {}
sequenceName = {}
trackPtMin = {}
trackPtMax = {}
doPlotsPCA = {}
numCutString = {}
denCutString = {}

selectedTracks = []

mainfolderName  ['generalTracks'] = 'Tracking/TrackParameters/generalTracks'
vertexfolderName['generalTracks'] = 'Tracking/PrimaryVertices/generalTracks'
trackPtMin      ['generalTracks'] = cms.double(0.)
trackPtMax      ['generalTracks'] = cms.double(100.)
doPlotsPCA      ['generalTracks'] = cms.bool(False)
numCutString    ['generalTracks'] = cms.string("")
denCutString    ['generalTracks'] = cms.string("")

trackSelector = cms.EDFilter('TrackSelector',
    src = cms.InputTag('generalTracks'),
    cut = cms.string("")
)

### highpurity definition: https://cmssdt.cern.ch/SDT/lxr/source/RecoTracker/FinalTrackSelectors/python/selectHighPurity_cfi.py
highPurityPtRange0to1 = trackSelector.clone()
highPurityPtRange0to1.cut = cms.string("quality('highPurity') & pt >= 0 & pt < 1 ")

sequenceName    ['highPurityPtRange0to1'] = cms.Sequence(highPurityPtRange0to1)
mainfolderName  ['highPurityPtRange0to1'] = 'Tracking/TrackParameters/highPurityTracks/pt_0to1'
vertexfolderName['highPurityPtRange0to1'] = 'Tracking/PrimaryVertices/highPurityTracks/pt_0to1'
trackPtMin      ['highPurityPtRange0to1'] = cms.double(0.)
trackPtMax      ['highPurityPtRange0to1'] = cms.double(1.)
numCutString    ['highPurityPtRange0to1'] = cms.string("") # default: " pt >= 1 & quality('highPurity') "
denCutString    ['highPurityPtRange0to1'] = cms.string(" pt >= 0 & pt < 1 ") # it is as in the default config (just be sure)

highPurityPtRange1to10 = trackSelector.clone()
highPurityPtRange1to10.cut = cms.string("quality('highPurity') & pt >= 1 & pt < 10 ")

sequenceName    ['highPurityPtRange1to10'] = cms.Sequence( highPurityPtRange1to10 )
mainfolderName  ['highPurityPtRange1to10'] = 'Tracking/TrackParameters/highPurityTracks/pt_1to10'
vertexfolderName['highPurityPtRange1to10'] = 'Tracking/PrimaryVertices/highPurityTracks/pt_1to10'
trackPtMin      ['highPurityPtRange1to10'] = cms.double(1.)
trackPtMax      ['highPurityPtRange1to10'] = cms.double(10.)
numCutString    ['highPurityPtRange1to10'] = cms.string("") # default: " pt >= 1 & quality('highPurity') "
denCutString    ['highPurityPtRange1to10'] = cms.string(" pt >= 1 & pt < 10 ") # it is as in the default config (just be sure)


highPurityPt10 = trackSelector.clone()
highPurityPt10.cut = cms.string("quality('highPurity') & pt >= 10")

sequenceName    ['highPurityPt10'] = cms.Sequence( highPurityPt10 )
mainfolderName  ['highPurityPt10'] = 'Tracking/TrackParameters/highPurityTracks/pt_10'
vertexfolderName['highPurityPt10'] = 'Tracking/PrimaryVertices/highPurityTracks/pt_10'
trackPtMin      ['highPurityPt10'] = cms.double(10.)
trackPtMax      ['highPurityPt10'] = cms.double(110.)
numCutString    ['highPurityPt10'] = cms.string("") # default: " pt >= 1 & quality('highPurity') "
denCutString    ['highPurityPt10'] = cms.string(" pt >= 10 ") # it is as in the default config (just be sure)


###### old monitored track collections
highPurityPt1 = trackSelector.clone()
highPurityPt1.cut = cms.string("quality('highPurity') & pt >= 1")

sequenceName    ['highPurityPt1'] = cms.Sequence(highPurityPt1)
mainfolderName  ['highPurityPt1'] = 'Tracking/TrackParameters/highPurityTracks/pt_1'
vertexfolderName['highPurityPt1'] = 'Tracking/PrimaryVertices/highPurityTracks/pt_1'
trackPtMin      ['highPurityPt1'] = cms.double(0.)
trackPtMax      ['highPurityPt1'] = cms.double(100.)
doPlotsPCA      ['highPurityPt1'] = cms.bool(True)
numCutString    ['highPurityPt1'] = cms.string("") # default: " pt >= 1 & quality('highPurity') "
denCutString    ['highPurityPt1'] = cms.string(" pt >= 1 ") # it is as in the default config (just be sure)

selectedTracks.extend( ['generalTracks'] )
#selectedTracks.extend( ['highPurityPtRange0to1']  )
#selectedTracks.extend( ['highPurityPtRange1to10'] )
#selectedTracks.extend( ['highPurityPt10']         )

selectedTracks.extend( ['highPurityPt1'] )

#selectedTracks2runSequence=cms.Sequence()
#for tracks in selectedTracks :
#    if tracks != 'generalTracks':
#        selectedTracks2runSequence+=sequenceName[tracks]

