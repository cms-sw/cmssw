import FWCore.ParameterSet.Config as cms

process = cms.Process('Analysis')

###################################################################
# import of standard configurations
###################################################################
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

###################################################################
# Configure the Global Tag
###################################################################
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '140X_dataRun3_Prompt_v2', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100000) )

###################################################################
# Set the process to run multi-threaded
###################################################################
process.options.numberOfThreads = 8

###################################################################
# Messages
###################################################################
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

###################################################################
# Source
###################################################################
#readFiles = cms.untracked.vstring(['root://eoscms.cern.ch//eos/cms/tier0/store/data/Run2024D/HLTPhysics/ALCARECO/TkAlV0s-PromptReco-v1/000/380/623/00000/0e0761c1-f437-4fca-b8b5-5793e7ab0748.root'])

import FWCore.Utilities.FileUtils as FileUtils
filelist = FileUtils.loadListFromFile("fileList.txt")
readFiles = cms.untracked.vstring( *filelist)

process.source = cms.Source("PoolSource",
                            fileNames = readFiles,
                            )

###################################################################
# TransientTrack from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideTransientTracks
###################################################################
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi')
process.load('TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff')

####################################################################
# Get the BeamSpot
####################################################################
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

####################################################################
# Track Refitter
####################################################################
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.k0shortRefitter = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    src = "ALCARECOTkAlKShortTracks",
    TrajectoryInEvent = True,
    NavigationSchool = '',
    TTRHBuilder = "WithAngleAndTemplate")

process.lambdaRefitter =  RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    src = "ALCARECOTkAlLambdaTracks",
    TrajectoryInEvent = True,
    NavigationSchool = '',
    TTRHBuilder = "WithAngleAndTemplate")

process.load("RecoVertex.V0Producer.generalV0Candidates_cfi")
import RecoVertex.V0Producer.generalV0Candidates_cfi
process.refittedKShorts = RecoVertex.V0Producer.generalV0Candidates_cfi.generalV0Candidates.clone(
    # which V0s to reconstruct
    doKShorts = True,
    doLambdas = False,
    # which TrackCollection to use for vertexing
    trackRecoAlgorithm = 'k0shortRefitter'
)

process.refittedLambdas = RecoVertex.V0Producer.generalV0Candidates_cfi.generalV0Candidates.clone(
    # which V0s to reconstruct
    doKShorts = False,
    doLambdas = True,
    # which TrackCollection to use for vertexing
    trackRecoAlgorithm = 'lambdaRefitter'
)

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",fileName=cms.string("TkAlV0Analysis.root"))

####################################################################
# Sequence
####################################################################
from DQM.TrackingMonitorSource.v0EventSelector_cfi import *
process.KShortEventSelector = v0EventSelector.clone(
    vertexCompositeCandidates = "generalV0Candidates:Kshort"  
)

process.LambdaEventSelector = v0EventSelector.clone(
    vertexCompositeCandidates = "generalV0Candidates:Lambda"  
)

####################################################################
# Sequence for refitting and re-doing the V0s
####################################################################
process.seqTrackselRefitK0short = cms.Sequence(process.offlineBeamSpot*
                                               # in case NavigatioSchool is set !=''
                                               #process.MeasurementTrackerEvent*
                                               process.KShortEventSelector*
                                               process.k0shortRefitter*
                                               process.refittedKShorts)

process.seqTracksRefitLambda = cms.Sequence(process.offlineBeamSpot*
                                            process.LambdaEventSelector*
                                            process.lambdaRefitter*
                                            process.refittedLambdas)
                                               
####################################################################
# Monitoring modules
####################################################################
from Alignment.OfflineValidation.tkAlV0sAnalyzer_cfi import *

process.K0Analysis = tkAlV0sAnalyzer.clone(
    vertexCompositeCandidates = 'refittedKShorts:Kshort',
    tracks = 'k0shortRefitter',
    histoPSet = tkAlV0sAnalyzer.histoPSet.clone(
        massPSet = tkAlV0sAnalyzer.histoPSet.massPSet.clone(
            nbins = 100,
            xmin = 0.400,
            xmax = 0.600
        )
    )   
)

process.LambdaAnalysis = tkAlV0sAnalyzer.clone(
    vertexCompositeCandidates = 'refittedLambdas:Lambda',
    tracks = 'lambdaRefitter',
    histoPSet = tkAlV0sAnalyzer.histoPSet.clone(
        massPSet = tkAlV0sAnalyzer.histoPSet.massPSet.clone(
            nbins = 100,
            xmin = 1.07,
            xmax = 1.17
        )
    )
)

####################################################################
# Path
####################################################################
process.p1 = cms.Path(process.seqTrackselRefitK0short +
                      process.K0Analysis)

process.p2 = cms.Path(process.seqTracksRefitLambda +
                      process.LambdaAnalysis)
