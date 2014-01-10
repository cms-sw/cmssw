import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOCalMinBiasFilterForSiStripGains = copy.deepcopy(hltHighLevel)
ALCARECOCalMinBiasFilterForSiStripGains.HLTPaths = ['pathALCARECOSiStripCalMinBias']
ALCARECOCalMinBiasFilterForSiStripGains.throw = True ## dont throw on unknown path names
ALCARECOCalMinBiasFilterForSiStripGains.TriggerResultsTag = cms.InputTag("TriggerResults","","RECO")
#process.TkAlMinBiasFilterForBS.eventSetupPathsKey = 'pathALCARECOTkAlMinBias:RECO'
#ALCARECODtCalibHLTFilter.andOr = True ## choose logical OR between Triggerbits


# ------------------------------------------------------------------------------


# FIXME: are the following blocks needed?

#this block is there to solve issue related to SiStripQualityRcd
#process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
#process.load("CalibTracker.SiStripESProducers.fake.SiStripDetVOffFakeESSource_cfi")
#process.es_prefer_fakeSiStripDetVOff = cms.ESPrefer("SiStripDetVOffFakeESSource","siStripDetVOffFakeESSource")


#process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")



# ------------------------------------------------------------------------------
# This is the sequence for track refitting of the track saved by SiStripCalMinBias
# to have access to transient objects produced during RECO step and not saved
# FIXME: should change names to make sure that there is no interference with any other part of the code when loading this cff
# (since it will be loaded any time the AlCaReco definitions are loaded)


from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import *

# FIXME: the beam-spot should be kept in the AlCaReco (if not already there) and dropped from here
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.TrackProducer.TrackRefitters_cff import *


CalibrationTracksRefit = TrackRefitter.clone(src = cms.InputTag("CalibrationTracks"))
CalibrationTracks = AlignmentTrackSelector.clone(
    #    src = 'generalTracks',
    src = 'ALCARECOSiStripCalMinBias',
    filter = True,
    applyBasicCuts = True,
    ptMin = 0.8,
    nHitMin = 6,
    chi2nMax = 10.,
    )

# refit and BS can be dropped if done together with RECO.
# track filter can be moved in acalreco if no otehr users
ALCARECOTrackFilterRefit = cms.Sequence(CalibrationTracks +
                                        offlineBeamSpot +
                                        CalibrationTracksRefit )


# ------------------------------------------------------------------------------
# This is the module actually doing the calibration

# FIXME: the safest option would be to import the basic cfi from a place mantained by the developer
SiStripCalib = cms.EDAnalyzer("SiStripGainFromCalibTree",
                              OutputGains         = cms.string('Gains_ASCII.txt'),
                              Tracks              = cms.untracked.InputTag('CalibrationTracksRefit'),
                              AlgoMode            = cms.untracked.string('PCL'),

                              #Gain quality cuts
                              minNrEntries        = cms.untracked.double(25),
                              maxChi2OverNDF      = cms.untracked.double(9999999.0),
                              maxMPVError         = cms.untracked.double(25.0),
                              
                              #track/cluster quality cuts
                              minTrackMomentum    = cms.untracked.double(2),
                              maxNrStrips         = cms.untracked.uint32(8),
                              
                              Validation          = cms.untracked.bool(False),
                              OldGainRemoving     = cms.untracked.bool(False),
                              FirstSetOfConstants = cms.untracked.bool(True),
                              
                              CalibrationLevel    = cms.untracked.int32(0), # 0==APV, 1==Laser, 2==module
                              
                              InputFiles          = cms.vstring(),
                              
                              UseCalibration     = cms.untracked.bool(False),
                              calibrationPath    = cms.untracked.string(""),

                              SinceAppendMode     = cms.bool(True),
                              IOVMode             = cms.string('Job'),
                              Record              = cms.string('SiStripApvGainRcd'),
                              doStoreOnDB         = cms.bool(True),
                              )

SiStripCalib.FirstSetOfConstants = cms.untracked.bool(False)
SiStripCalib.CalibrationLevel    = cms.untracked.int32(0) # 0==APV, 1==Laser, 2==module


# ------------------------------------------------------------------------------
# Here we define additional services needed by the module

# FIXME: find a better place or remove (PoolDBOutputService is not required in the final config since the DB will be written in the following step
PoolDBOutputService = cms.Service("PoolDBOutputService",
                                  BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                  DBParameters = cms.PSet(
                                      messageLevel = cms.untracked.int32(2),
                                      authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
                                      ),
                                  timetype = cms.untracked.string('runnumber'),
                                  connect = cms.string('sqlite_file:Gains_Sqlite.db'),
                                  toPut = cms.VPSet(cms.PSet(
                                      record = cms.string('SiStripApvGainRcd'),
                                      tag = cms.string('IdealGainTag')
                                      ))
                                  )

TFileService = cms.Service("TFileService",
                           fileName = cms.string('Gains_Tree.root')  
                           )








seqALCARECOPromptCalibProdSiStripGains = cms.Sequence(ALCARECOCalMinBiasFilterForSiStripGains *
                                                      ALCARECOTrackFilterRefit *
                                                      SiStripCalib)


