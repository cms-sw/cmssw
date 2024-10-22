import FWCore.ParameterSet.Config as cms

# configures track finding s/w to use KF emulator instead of KF simulator
def newKFConfig(process):
  process.l1tTTTracksFromTrackletEmulation.Fakefit = True

# configures track finding s/w to behave as track finding f/w
def fwConfig(process):
  newKFConfig(process)
  process.TrackTriggerSetup.Firmware.FreqBE = 240 # Frequency of DTC & KF (determines truncation)
  process.l1tTTTracksFromTrackletEmulation.RemovalType = ""
  process.l1tTTTracksFromTrackletEmulation.DoMultipleMatches = False
  process.l1tTTTracksFromTrackletEmulation.StoreTrackBuilderOutput = True

# configures track finding s/w to behave as a subchain of processing steps
def reducedConfig(process):
  fwConfig(process)
  process.TrackTriggerSetup.KalmanFilter.NumWorker = 1
  process.ChannelAssignment.SeedTypes = cms.vstring( "L1L2" )
  process.ChannelAssignment.SeedTypesSeedLayers = cms.PSet( L1L2 = cms.vint32( 1,  2 ) )
  process.ChannelAssignment.SeedTypesProjectionLayers = cms.PSet( L1L2 = cms.vint32(  3,  4,  5,  6 ) )
  # this are tt::Setup::dtcId in order as in process.l1tTTTracksFromTrackletEmulation.processingModulesFile translated by 
  # reverssing naming logic described in L1FPGATrackProducer
  # TO DO: Eliminate cfg param IRChannelsIn by taking this info from Tracklet wiring map.
  process.ChannelAssignment.IRChannelsIn = cms.vint32( 0, 1, 25, 2, 26, 4, 5, 29, 6, 30, 7, 31, 8, 9, 33 )
  process.l1tTTTracksFromTrackletEmulation.Reduced = True
  process.l1tTTTracksFromTrackletEmulation.memoryModulesFile = 'L1Trigger/TrackFindingTracklet/data/reduced_memorymodules.dat'
  process.l1tTTTracksFromTrackletEmulation.processingModulesFile = 'L1Trigger/TrackFindingTracklet/data/reduced_processingmodules.dat'
  process.l1tTTTracksFromTrackletEmulation.wiresFile = 'L1Trigger/TrackFindingTracklet/data/reduced_wires.dat'
