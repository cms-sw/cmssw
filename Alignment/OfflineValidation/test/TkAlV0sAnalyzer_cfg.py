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
process.GlobalTag = GlobalTag(process.GlobalTag, '133X_mcRun3_2023_realistic_v3', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

###################################################################
# Messages
###################################################################
process.load('FWCore.MessageService.MessageLogger_cfi')   

###################################################################
# Source
###################################################################
readFiles = cms.untracked.vstring('file:../../../TkAlV0s.root')
process.source = cms.Source("PoolSource",
                            fileNames = readFiles,
                            #skipEvents = cms.untracked.uint32(45000)
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
process.TrackRefitter = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone()
process.TrackRefitter.src = "ALCARECOTkAlKShortTracks"
process.TrackRefitter.TrajectoryInEvent = True
process.TrackRefitter.NavigationSchool = ''
process.TrackRefitter.TTRHBuilder = "WithAngleAndTemplate"

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",fileName=cms.string("TkAlV0Analysis.root"))

####################################################################
# Sequence
####################################################################
process.seqTrackselRefit = cms.Sequence(process.offlineBeamSpot*
                                        # in case NavigatioSchool is set !=''
                                        #process.MeasurementTrackerEvent*
                                        process.TrackRefitter)

####################################################################
# Additional output definition
####################################################################
process.analysis = cms.EDAnalyzer('TkAlV0sAnalyzer',
                                  #tracks = cms.untracked.InputTag('TrackRefitter'))
                                  tracks = cms.untracked.InputTag('ALCARECOTkAlKShortTracks'))

####################################################################
# Path
####################################################################
process.p = cms.Path(#process.seqTrackselRefit +
                     process.analysis)
