import FWCore.ParameterSet.Config as cms

allTrackProducer   = {}
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
doRecHitVsPtVsEtaPerTrack           = {}
doGoodTrackRecHitVsPhiVsEtaPerTrack = {}
doLayersVsPhiVsEtaPerTrack          = {}
doGoodTrackLayersVsPhiVsEtaPerTrack = {}
doPUmonitoring                      = {}
doPlotsVsBXlumi                     = {}
doPlotsVsGoodPVtx                   = {}
doEffFromHitPatternVsPU             = {}
doEffFromHitPatternVsBX             = {}
doEffFromHitPatternVsLumi           = {}
doStopSource                        = {}

selectedTracks = []

allTrackProducer['generalTracks'] = 'generalTracks'
mainfolderName  ['generalTracks'] = 'Tracking/TrackParameters/generalTracks'
vertexfolderName['generalTracks'] = 'Tracking/PrimaryVertices/generalTracks'
trackPtN        ['generalTracks'] = cms.int32(100)
trackPtMin      ['generalTracks'] = cms.double(0.)
trackPtMax      ['generalTracks'] = cms.double(100.)
doPlotsPCA      ['generalTracks'] = cms.bool(False)
numCutString    ['generalTracks'] = cms.string("quality('highPurity')") # num := den + quality('highPurity')
denCutString    ['generalTracks'] = cms.string("") # den := kinematics cuts
doGoodTracksPlots                   ['generalTracks'] = cms.bool(True)
doTrackerSpecific                   ['generalTracks'] = cms.bool(True)
doHitPropertiesPlots                ['generalTracks'] = cms.bool(True)
doGeneralPropertiesPlots            ['generalTracks'] = cms.bool(True)
doBeamSpotPlots                     ['generalTracks'] = cms.bool(True)
doSeedParameterHistos               ['generalTracks'] = cms.bool(False)
doRecHitVsPhiVsEtaPerTrack          ['generalTracks'] = cms.bool(True)
doRecHitVsPtVsEtaPerTrack           ['generalTracks'] = cms.bool(True)
doGoodTrackRecHitVsPhiVsEtaPerTrack ['generalTracks'] = cms.bool(True)
doLayersVsPhiVsEtaPerTrack          ['generalTracks'] = cms.bool(True)
doGoodTrackLayersVsPhiVsEtaPerTrack ['generalTracks'] = cms.bool(True)
doPUmonitoring                      ['generalTracks'] = cms.bool(False)
doPlotsVsBXlumi                     ['generalTracks'] = cms.bool(False)
doPlotsVsGoodPVtx                   ['generalTracks'] = cms.bool(True)
doEffFromHitPatternVsPU             ['generalTracks'] = cms.bool(True)
doEffFromHitPatternVsBX             ['generalTracks'] = cms.bool(False)
doEffFromHitPatternVsLumi           ['generalTracks'] = cms.bool(False)
doStopSource                        ['generalTracks'] = cms.bool(True)

trackSelector = cms.EDFilter('TrackSelector',
    src = cms.InputTag('generalTracks'),
    cut = cms.string("")
)

### highpurity definition: https://cmssdt.cern.ch/SDT/lxr/source/RecoTracker/FinalTrackSelectors/python/selectHighPurity_cfi.py
highPurityPtRange0to1 = trackSelector.clone()
highPurityPtRange0to1.cut = cms.string("quality('highPurity') & pt >= 0 & pt < 1 ")

sequenceName    ['highPurityPtRange0to1'] = highPurityPtRange0to1
allTrackProducer['highPurityPtRange0to1'] = 'generalTracks'
mainfolderName  ['highPurityPtRange0to1'] = 'Tracking/TrackParameters/highPurityTracks/pt_0to1'
vertexfolderName['highPurityPtRange0to1'] = 'Tracking/PrimaryVertices/highPurityTracks/pt_0to1'
trackPtN        ['highPurityPtRange0to1'] = cms.int32(10)
trackPtMin      ['highPurityPtRange0to1'] = cms.double(0.)
trackPtMax      ['highPurityPtRange0to1'] = cms.double(1.)
numCutString    ['highPurityPtRange0to1'] = cms.string(" pt >= 0 & pt < 1 & quality('highPurity')") # num := den + quality('highPurity') [it is the same as the main selection, but just to be sure ...]
denCutString    ['highPurityPtRange0to1'] = cms.string(" pt >= 0 & pt < 1 ") # den := kinematics cut
doPlotsPCA      ['highPurityPtRange0to1'] = cms.bool(False)
doGoodTracksPlots                   ['highPurityPtRange0to1'] = cms.bool(False)
doTrackerSpecific                   ['highPurityPtRange0to1'] = cms.bool(False)
doHitPropertiesPlots                ['highPurityPtRange0to1'] = cms.bool(True)
doGeneralPropertiesPlots            ['highPurityPtRange0to1'] = cms.bool(True)
doBeamSpotPlots                     ['highPurityPtRange0to1'] = cms.bool(True)
doSeedParameterHistos               ['highPurityPtRange0to1'] = cms.bool(False)
doRecHitVsPhiVsEtaPerTrack          ['highPurityPtRange0to1'] = cms.bool(True)
doRecHitVsPtVsEtaPerTrack           ['highPurityPtRange0to1'] = cms.bool(True)
doGoodTrackRecHitVsPhiVsEtaPerTrack ['highPurityPtRange0to1'] = cms.bool(False)
doLayersVsPhiVsEtaPerTrack          ['highPurityPtRange0to1'] = cms.bool(True)
doGoodTrackLayersVsPhiVsEtaPerTrack ['highPurityPtRange0to1'] = cms.bool(False)
doPUmonitoring                      ['highPurityPtRange0to1'] = cms.bool(True)
doPlotsVsBXlumi                     ['highPurityPtRange0to1'] = cms.bool(False)
doPlotsVsGoodPVtx                   ['highPurityPtRange0to1'] = cms.bool(True)
doEffFromHitPatternVsPU             ['highPurityPtRange0to1'] = cms.bool(False)
doEffFromHitPatternVsBX             ['highPurityPtRange0to1'] = cms.bool(False)
doEffFromHitPatternVsLumi           ['highPurityPtRange0to1'] = cms.bool(False)
doStopSource                        ['highPurityPtRange0to1'] = cms.bool(True)

highPurityPtRange1to10 = trackSelector.clone()
highPurityPtRange1to10.cut = cms.string("quality('highPurity') & pt >= 1 & pt < 10 ")

sequenceName    ['highPurityPtRange1to10'] = highPurityPtRange1to10 
allTrackProducer['highPurityPtRange1to10'] = 'generalTracks'
mainfolderName  ['highPurityPtRange1to10'] = 'Tracking/TrackParameters/highPurityTracks/pt_1to10'
vertexfolderName['highPurityPtRange1to10'] = 'Tracking/PrimaryVertices/highPurityTracks/pt_1to10'
trackPtN        ['highPurityPtRange1to10'] = cms.int32(10)
trackPtMin      ['highPurityPtRange1to10'] = cms.double(1.)
trackPtMax      ['highPurityPtRange1to10'] = cms.double(10.)
numCutString    ['highPurityPtRange1to10'] = cms.string(" pt >= 1 & pt < 10 & quality('highPurity')") # num := den + quality('highPurity') [it is the same as the main selection, but just to be sure ...]
denCutString    ['highPurityPtRange1to10'] = cms.string(" pt >= 1 & pt < 10 ") # den := kinematics cuts
doGoodTracksPlots                   ['highPurityPtRange1to10'] = cms.bool(True)
doTrackerSpecific                   ['highPurityPtRange1to10'] = cms.bool(True)
doHitPropertiesPlots                ['highPurityPtRange1to10'] = cms.bool(True)
doGeneralPropertiesPlots            ['highPurityPtRange1to10'] = cms.bool(True)
doBeamSpotPlots                     ['highPurityPtRange1to10'] = cms.bool(True)
doSeedParameterHistos               ['highPurityPtRange1to10'] = cms.bool(False)
doRecHitVsPhiVsEtaPerTrack          ['highPurityPtRange1to10'] = cms.bool(True)
doRecHitVsPtVsEtaPerTrack           ['highPurityPtRange1to10'] = cms.bool(True)
doGoodTrackRecHitVsPhiVsEtaPerTrack ['highPurityPtRange1to10'] = cms.bool(True)
doLayersVsPhiVsEtaPerTrack          ['highPurityPtRange1to10'] = cms.bool(True)
doGoodTrackLayersVsPhiVsEtaPerTrack ['highPurityPtRange1to10'] = cms.bool(True)
doPUmonitoring                      ['highPurityPtRange1to10'] = cms.bool(False)
doPlotsVsBXlumi                     ['highPurityPtRange1to10'] = cms.bool(False)
doPlotsVsGoodPVtx                   ['highPurityPtRange1to10'] = cms.bool(True)
doEffFromHitPatternVsPU             ['highPurityPtRange1to10'] = cms.bool(False)
doEffFromHitPatternVsBX             ['highPurityPtRange1to10'] = cms.bool(False)
doEffFromHitPatternVsLumi           ['highPurityPtRange1to10'] = cms.bool(False)
doStopSource                        ['highPurityPtRange1to10'] = cms.bool(True)

highPurityPt10 = trackSelector.clone()
highPurityPt10.cut = cms.string("quality('highPurity') & pt >= 10")

sequenceName    ['highPurityPt10'] = highPurityPt10 
allTrackProducer['highPurityPt10'] = 'generalTracks'
mainfolderName  ['highPurityPt10'] = 'Tracking/TrackParameters/highPurityTracks/pt_10'
vertexfolderName['highPurityPt10'] = 'Tracking/PrimaryVertices/highPurityTracks/pt_10'
trackPtN        ['highPurityPt10'] = cms.int32(100)
trackPtMin      ['highPurityPt10'] = cms.double(10.)
trackPtMax      ['highPurityPt10'] = cms.double(110.)
numCutString    ['highPurityPt10'] = cms.string(" pt >= 10 & quality('highPurity')") # num := den + quality('highPurity') [it is the same as the main selection, but just to be sure ...]
denCutString    ['highPurityPt10'] = cms.string(" pt >= 10 ") # den := kinematics cuts
doGoodTracksPlots                   ['highPurityPt10'] = cms.bool(True)
doTrackerSpecific                   ['highPurityPt10'] = cms.bool(True)
doHitPropertiesPlots                ['highPurityPt10'] = cms.bool(True)
doGeneralPropertiesPlots            ['highPurityPt10'] = cms.bool(True)
doBeamSpotPlots                     ['highPurityPt10'] = cms.bool(True)
doSeedParameterHistos               ['highPurityPt10'] = cms.bool(False)
doRecHitVsPhiVsEtaPerTrack          ['highPurityPt10'] = cms.bool(True)
doRecHitVsPtVsEtaPerTrack           ['highPurityPt10'] = cms.bool(True)
doGoodTrackRecHitVsPhiVsEtaPerTrack ['highPurityPt10'] = cms.bool(True)
doLayersVsPhiVsEtaPerTrack          ['highPurityPt10'] = cms.bool(True)
doGoodTrackLayersVsPhiVsEtaPerTrack ['highPurityPt10'] = cms.bool(True)
doPUmonitoring                      ['highPurityPt10'] = cms.bool(False)
doPlotsVsBXlumi                     ['highPurityPt10'] = cms.bool(False)
doPlotsVsGoodPVtx                   ['highPurityPt10'] = cms.bool(True)
doEffFromHitPatternVsPU             ['highPurityPt10'] = cms.bool(False)
doEffFromHitPatternVsBX             ['highPurityPt10'] = cms.bool(False)
doEffFromHitPatternVsLumi           ['highPurityPt10'] = cms.bool(False)
doStopSource                        ['highPurityPt10'] = cms.bool(True)

###### old monitored track collections
highPurityPt1 = trackSelector.clone()
highPurityPt1.cut = cms.string("quality('highPurity') & pt >= 1")

sequenceName    ['highPurityPt1'] = highPurityPt1
allTrackProducer['highPurityPt1'] = 'generalTracks'
mainfolderName  ['highPurityPt1'] = 'Tracking/TrackParameters/highPurityTracks/pt_1'
vertexfolderName['highPurityPt1'] = 'Tracking/PrimaryVertices/highPurityTracks/pt_1'
trackPtN        ['highPurityPt1'] = cms.int32(100)
trackPtMin      ['highPurityPt1'] = cms.double(0.)
trackPtMax      ['highPurityPt1'] = cms.double(100.)
doPlotsPCA      ['highPurityPt1'] = cms.bool(True)
numCutString    ['highPurityPt1'] = cms.string(" pt >= 1 & quality('highPurity')") # num := den + quality('highPurity') [it is the same as the main selection, but just to be sure ...]
denCutString    ['highPurityPt1'] = cms.string(" pt >= 1 ") # den := kinematics cut
doGoodTracksPlots                   ['highPurityPt1'] = cms.bool(True)
doTrackerSpecific                   ['highPurityPt1'] = cms.bool(True)
doHitPropertiesPlots                ['highPurityPt1'] = cms.bool(True)
doGeneralPropertiesPlots            ['highPurityPt1'] = cms.bool(True)
doBeamSpotPlots                     ['highPurityPt1'] = cms.bool(True)
doSeedParameterHistos               ['highPurityPt1'] = cms.bool(False)
doRecHitVsPhiVsEtaPerTrack          ['highPurityPt1'] = cms.bool(True)
doRecHitVsPtVsEtaPerTrack           ['highPurityPt1'] = cms.bool(True)
doGoodTrackRecHitVsPhiVsEtaPerTrack ['highPurityPt1'] = cms.bool(True)
doLayersVsPhiVsEtaPerTrack          ['highPurityPt1'] = cms.bool(True)
doGoodTrackLayersVsPhiVsEtaPerTrack ['highPurityPt1'] = cms.bool(True)
doPUmonitoring                      ['highPurityPt1'] = cms.bool(False)
doPlotsVsBXlumi                     ['highPurityPt1'] = cms.bool(False)
doPlotsVsGoodPVtx                   ['highPurityPt1'] = cms.bool(True)
doEffFromHitPatternVsPU             ['highPurityPt1'] = cms.bool(True)
doEffFromHitPatternVsBX             ['highPurityPt1'] = cms.bool(True)
doEffFromHitPatternVsLumi           ['highPurityPt1'] = cms.bool(True)
doStopSource                        ['highPurityPt1'] = cms.bool(True)

###### forward-only monitored track collections
highPurityPt1Eta2p5to3p0 = trackSelector.clone()
highPurityPt1Eta2p5to3p0.cut = cms.string("quality('highPurity') & pt >= 1 & abs(eta) > 2.5")

sequenceName    ['highPurityPt1Eta2p5to3p0'] = highPurityPt1Eta2p5to3p0
allTrackProducer['highPurityPt1Eta2p5to3p0'] = 'generalTracks'
mainfolderName  ['highPurityPt1Eta2p5to3p0'] = 'Tracking/TrackParameters/highPurityTracks/pt_1_Eta_2p5'
vertexfolderName['highPurityPt1Eta2p5to3p0'] = 'Tracking/PrimaryVertices/highPurityTracks/pt_1_Eta_2p5'
trackPtN        ['highPurityPt1Eta2p5to3p0'] = cms.int32(100)
trackPtMin	['highPurityPt1Eta2p5to3p0'] = cms.double(0.)
trackPtMax	['highPurityPt1Eta2p5to3p0'] = cms.double(100.)
doPlotsPCA	['highPurityPt1Eta2p5to3p0'] = cms.bool(True)
numCutString    ['highPurityPt1Eta2p5to3p0'] = cms.string(" pt >= 1 & abs(eta) > 2.5 & quality('highPurity')") # num := den + quality('highPurity') [it is the same as the main selection, but just to be sure ...]
denCutString    ['highPurityPt1Eta2p5to3p0'] = cms.string(" pt >= 1 & abs(eta) > 2.5") # den := kinematics cut
doGoodTracksPlots                   ['highPurityPt1Eta2p5to3p0'] = cms.bool(True)
doTrackerSpecific                   ['highPurityPt1Eta2p5to3p0'] = cms.bool(True)
doHitPropertiesPlots                ['highPurityPt1Eta2p5to3p0'] = cms.bool(True)
doGeneralPropertiesPlots            ['highPurityPt1Eta2p5to3p0'] = cms.bool(True)
doBeamSpotPlots                     ['highPurityPt1Eta2p5to3p0'] = cms.bool(True)
doSeedParameterHistos               ['highPurityPt1Eta2p5to3p0'] = cms.bool(False)
doRecHitVsPhiVsEtaPerTrack          ['highPurityPt1Eta2p5to3p0'] = cms.bool(True)
doRecHitVsPtVsEtaPerTrack           ['highPurityPt1Eta2p5to3p0'] = cms.bool(True)
doGoodTrackRecHitVsPhiVsEtaPerTrack ['highPurityPt1Eta2p5to3p0'] = cms.bool(True)
doLayersVsPhiVsEtaPerTrack          ['highPurityPt1Eta2p5to3p0'] = cms.bool(True)
doGoodTrackLayersVsPhiVsEtaPerTrack ['highPurityPt1Eta2p5to3p0'] = cms.bool(True)
doPUmonitoring                      ['highPurityPt1Eta2p5to3p0'] = cms.bool(False)
doPlotsVsBXlumi                     ['highPurityPt1Eta2p5to3p0'] = cms.bool(False)
doPlotsVsGoodPVtx                   ['highPurityPt1Eta2p5to3p0'] = cms.bool(True)
doEffFromHitPatternVsPU             ['highPurityPt1Eta2p5to3p0'] = cms.bool(True)
doEffFromHitPatternVsBX             ['highPurityPt1Eta2p5to3p0'] = cms.bool(False)
doEffFromHitPatternVsLumi           ['highPurityPt1Eta2p5to3p0'] = cms.bool(False)
doStopSource                        ['highPurityPt1Eta2p5to3p0'] = cms.bool(True)




###### all tracks (no pt cut) associated to the PV
###### association is dz<1mm 
from CommonTools.RecoAlgos.TrackWithVertexSelector_cfi import *

trackAssociated2pvSelector = trackWithVertexSelector.clone()
# the track collection
trackAssociated2pvSelector.src = cms.InputTag('generalTracks')
# kinematic cuts  (pT in GeV)
trackAssociated2pvSelector.etaMin = cms.double(0.0)
trackAssociated2pvSelector.etaMax = cms.double(5.0)
trackAssociated2pvSelector.ptMin = cms.double(0.0)
trackAssociated2pvSelector.ptMax = cms.double(100000.0)
# impact parameter cut (in cm)
trackAssociated2pvSelector.d0Max = cms.double(999.)
trackAssociated2pvSelector.dzMax = cms.double(999.)
# quality cuts (valid hits, normalized chi2)
trackAssociated2pvSelector.normalizedChi2         = cms.double(999999.)
trackAssociated2pvSelector.numberOfValidHits      = cms.uint32(0)
trackAssociated2pvSelector.numberOfLostHits       = cms.uint32(999) ## at most 999 lost hits
trackAssociated2pvSelector.numberOfValidPixelHits = cms.uint32(0) ## at least <n> hits in the pixels
trackAssociated2pvSelector.ptErrorCut             = cms.double(9999999.) ## [pTError/pT]*max(1,normChi2) <= ptErrorCut
trackAssociated2pvSelector.quality = cms.string("highPurity") # quality cut as defined in reco::TrackBase
# compatibility with a vertex ?
trackAssociated2pvSelector.useVtx       = cms.bool(True)
trackAssociated2pvSelector.vertexTag    = cms.InputTag('trackingDQMgoodOfflinePrimaryVertices')
trackAssociated2pvSelector.timesTag     = cms.InputTag('')
trackAssociated2pvSelector.timeResosTag = cms.InputTag('')
trackAssociated2pvSelector.nVertices    = cms.uint32(1) ## how many vertices to look at before dropping the track
trackAssociated2pvSelector.vtxFallback  = cms.bool(True) ## falback to beam spot if there are no vertices
# uses vtx=(0,0,0) with deltaZeta=15.9, deltaRho = 0.2
trackAssociated2pvSelector.zetaVtx        = cms.double(999.)
#trackAssociated2pvSelector.rhoVtx         = cms.double(0.2) ## tags used by b-tagging folks
trackAssociated2pvSelector.rhoVtx         = cms.double(999.) ## tags used by b-tagging folks
trackAssociated2pvSelector.nSigmaDtVertex = cms.double(0)
# should _not_ be used for the TrackWithVertexRefSelector
trackAssociated2pvSelector.copyExtras       = cms.untracked.bool(False) ## copies also extras and rechits on RECO
trackAssociated2pvSelector.copyTrajectories = cms.untracked.bool(False) # don't set this to true on AOD!

highPurityPV0p1 = trackAssociated2pvSelector.clone()
highPurityPV0p1.zetaVtx = cms.double(0.1) # wrt PV
#highPurityPV0p1.dzMax   = cms.double(0.1) # wrt BS

PV0p1 = highPurityPV0p1.clone()
PV0p1.quality = cms.string("") # quality cut as defined in reco::TrackBase

#sequenceName    ['highPurityPV0p1'] = highPurityPV0p1
sequenceName    ['highPurityPV0p1'] = highPurityPV0p1+PV0p1
allTrackProducer['highPurityPV0p1'] = 'PV0p1'
mainfolderName  ['highPurityPV0p1'] = 'Tracking/TrackParameters/highPurityTracks/dzPV0p1'
vertexfolderName['highPurityPV0p1'] = 'Tracking/PrimaryVertices/highPurityTracks/dzPV0p1'
trackPtN        ['highPurityPV0p1'] = cms.int32(100)
trackPtMin      ['highPurityPV0p1'] = cms.double(0.)
trackPtMax      ['highPurityPV0p1'] = cms.double(100.)
doPlotsPCA      ['highPurityPV0p1'] = cms.bool(True)
numCutString    ['highPurityPV0p1'] = cms.string("quality('highPurity')") # num := den + quality('highPurity')
denCutString    ['highPurityPV0p1'] = cms.string("") # den := kinematic cuts
doGoodTracksPlots                   ['highPurityPV0p1'] = cms.bool(True)
doTrackerSpecific                   ['highPurityPV0p1'] = cms.bool(True)
doHitPropertiesPlots                ['highPurityPV0p1'] = cms.bool(True)
doGeneralPropertiesPlots            ['highPurityPV0p1'] = cms.bool(True)
doBeamSpotPlots                     ['highPurityPV0p1'] = cms.bool(True)
doSeedParameterHistos               ['highPurityPV0p1'] = cms.bool(False)
doRecHitVsPhiVsEtaPerTrack          ['highPurityPV0p1'] = cms.bool(True)
doRecHitVsPtVsEtaPerTrack           ['highPurityPV0p1'] = cms.bool(True)
doGoodTrackRecHitVsPhiVsEtaPerTrack ['highPurityPV0p1'] = cms.bool(True)
doLayersVsPhiVsEtaPerTrack          ['highPurityPV0p1'] = cms.bool(True)
doGoodTrackLayersVsPhiVsEtaPerTrack ['highPurityPV0p1'] = cms.bool(True)
doPUmonitoring                      ['highPurityPV0p1'] = cms.bool(False)
doPlotsVsBXlumi                     ['highPurityPV0p1'] = cms.bool(False)
doPlotsVsGoodPVtx                   ['highPurityPV0p1'] = cms.bool(True)
doEffFromHitPatternVsPU             ['highPurityPV0p1'] = cms.bool(True)
doEffFromHitPatternVsBX             ['highPurityPV0p1'] = cms.bool(False)
doEffFromHitPatternVsLumi           ['highPurityPV0p1'] = cms.bool(True)
doStopSource                        ['highPurityPV0p1'] = cms.bool(True)

#pixel tracks
hiConformalPixelTracksQA = trackSelector.clone()
hiConformalPixelTracksQA.src = cms.InputTag('hiConformalPixelTracks')
hiConformalPixelTracksQA.cut = cms.string("chi2/ndof/hitPattern.trackerLayersWithMeasurement < 200")

sequenceName    ['hiConformalPixelTracksQA'] = hiConformalPixelTracksQA
allTrackProducer['hiConformalPixelTracksQA'] = 'generalTracks'
mainfolderName  ['hiConformalPixelTracksQA'] = 'Tracking/TrackParameters/hiConformalPixelTracks'
vertexfolderName['hiConformalPixelTracksQA'] = 'Tracking/PrimaryVertices/hiConformalPixelTracks'
trackPtN        ['hiConformalPixelTracksQA'] = cms.int32(100)
trackPtMin      ['hiConformalPixelTracksQA'] = cms.double(0.)
trackPtMax      ['hiConformalPixelTracksQA'] = cms.double(10.)
numCutString    ['hiConformalPixelTracksQA'] = cms.string(" pt >= 0 ") 
denCutString    ['hiConformalPixelTracksQA'] = cms.string(" pt >= 0 ") 
doPlotsPCA      ['hiConformalPixelTracksQA'] = cms.bool(False)
doGoodTracksPlots                   ['hiConformalPixelTracksQA'] = cms.bool(False)
doTrackerSpecific                   ['hiConformalPixelTracksQA'] = cms.bool(False)
doHitPropertiesPlots                ['hiConformalPixelTracksQA'] = cms.bool(True)
doGeneralPropertiesPlots            ['hiConformalPixelTracksQA'] = cms.bool(True)
doBeamSpotPlots                     ['hiConformalPixelTracksQA'] = cms.bool(True)
doSeedParameterHistos               ['hiConformalPixelTracksQA'] = cms.bool(False)
doRecHitVsPhiVsEtaPerTrack          ['hiConformalPixelTracksQA'] = cms.bool(True)
doRecHitVsPtVsEtaPerTrack           ['hiConformalPixelTracksQA'] = cms.bool(True)
doGoodTrackRecHitVsPhiVsEtaPerTrack ['hiConformalPixelTracksQA'] = cms.bool(False)
doLayersVsPhiVsEtaPerTrack          ['hiConformalPixelTracksQA'] = cms.bool(True)
doGoodTrackLayersVsPhiVsEtaPerTrack ['hiConformalPixelTracksQA'] = cms.bool(False)
doPUmonitoring                      ['hiConformalPixelTracksQA'] = cms.bool(True)
doPlotsVsBXlumi                     ['hiConformalPixelTracksQA'] = cms.bool(False)
doPlotsVsGoodPVtx                   ['hiConformalPixelTracksQA'] = cms.bool(True)
doEffFromHitPatternVsPU             ['hiConformalPixelTracksQA'] = cms.bool(False)
doEffFromHitPatternVsBX             ['hiConformalPixelTracksQA'] = cms.bool(False)
doEffFromHitPatternVsLumi           ['hiConformalPixelTracksQA'] = cms.bool(False)
doStopSource                        ['hiConformalPixelTracksQA'] = cms.bool(True)

selectedTracks.extend( ['generalTracks'] )
#selectedTracks.extend( ['highPurityPtRange0to1']  )
#selectedTracks.extend( ['highPurityPtRange1to10'] )
#selectedTracks.extend( ['highPurityPt10']         )

selectedTracks.extend( ['highPurityPt1'] )
selectedTracks.extend( ['highPurityPtRange0to1'] )
selectedTracks.extend( ['highPurityPV0p1'] )

# not by default
#selectedTracks.extend( ['highPurityPt1Eta2p5to3p0'] )


#selectedTracks2runSequence=cms.Sequence()
#for tracks in selectedTracks :
#    if tracks != 'generalTracks':
#        selectedTracks2runSequence+=sequenceName[tracks]

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(selectedTracks, func=lambda selectedTracks: selectedTracks.extend( ['hiConformalPixelTracksQA'] ))
