import FWCore.ParameterSet.Config as cms

# configures track finding s/w to use KF emulator instead of KF simulator
def newKFConfig(process):
  process.TTTracksFromTrackletEmulation.Fakefit = True

# configures track finding s/w to behave as track finding f/w
def fwConfig(process):
  newKFConfig(process)
  process.TrackTriggerSetup.Firmware.FreqBE = 240 # Frequency of DTC & KF (determines truncation)
  process.TTTracksFromTrackletEmulation.RemovalType = ""
  process.TTTracksFromTrackletEmulation.DoMultipleMatches = False
  process.TTTracksFromTrackletEmulation.StoreTrackBuilderOutput = True
  process.ChannelAssignment.UseDuplicateRemoval = False
  process.TrackTriggerSetup.KalmanFilter.NumWorker = 8

# configures track finding s/w to behave as a subchain of processing steps
def reducedConfig(process):
  fwConfig(process)
  process.TrackTriggerSetup.KalmanFilter.NumWorker = 1
  process.ChannelAssignment.SeedTypes = cms.vstring( "L1L2" )
  process.ChannelAssignment.SeedTypesSeedLayers = cms.PSet( L1L2 = cms.vint32( 1,  2 ) )
  process.ChannelAssignment.SeedTypesProjectionLayers = cms.PSet( L1L2 = cms.vint32(  3,  4,  5,  6 ) )
  # this are tt::Setup::dtcId in order as in process.TTTracksFromTrackletEmulation.processingModulesFile translated by 
  # reverssing naming logic described in L1FPGATrackProducer
  # TO DO: Eliminate cfg param IRChannelsIn by taking this info from Tracklet wiring map.
  process.ChannelAssignment.IRChannelsIn = cms.vint32( 0, 1, 25, 2, 26, 4, 28, 5, 29, 6, 30, 7, 31, 8, 32, 9, 33 )
  process.TTTracksFromTrackletEmulation.Reduced = True
  process.TTTracksFromTrackletEmulation.memoryModulesFile = 'L1Trigger/TrackFindingTracklet/data/reduced_memorymodules.dat'
  process.TTTracksFromTrackletEmulation.processingModulesFile = 'L1Trigger/TrackFindingTracklet/data/reduced_processingmodules.dat'
  process.TTTracksFromTrackletEmulation.wiresFile = 'L1Trigger/TrackFindingTracklet/data/reduced_wires.dat'
