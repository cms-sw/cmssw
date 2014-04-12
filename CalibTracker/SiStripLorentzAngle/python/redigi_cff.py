import FWCore.ParameterSet.Config as cms

dbfile = 'sqlite_file:SiStripLorentzAngle_CalibrationEnsemble.db'
dbtag = 'SiStripLorentzAngle_CalibrationEnsemble_31X'

#digitization
from SimTracker.SiStripDigitizer.SiStripDigi_APVModeDec_cff import *
from SimGeneral.Configuration.SimGeneral_cff import *
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
pdigi = cms.Sequence(cms.SequencePlaceholder("randomEngineStateProducer")*cms.SequencePlaceholder("mix")*
                     simSiStripDigis*trackingParticles*SiStripDigiToRaw)
simSiStripDigis.chargeDivisionsPerStrip = 100

#Digitize and reconstruct using smeared values of mobility
SiStripLorentzAngle = cms.ESSource("PoolDBESSource",
                                   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                   DBParameters = cms.PSet( messageLevel = cms.untracked.int32(0),
                                                            authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
                                                            ),
                                   timetype = cms.untracked.string('runnumber'),
                                   connect = cms.string(dbfile),
                                   toGet = cms.VPSet(
    cms.PSet( record = cms.string('SiStripLorentzAngleRcd'), tag = cms.string(dbtag) )
    )
                                   )
es_prefer_SiStripLorentzAngle = cms.ESPrefer("PoolDBESSource","SiStripLorentzAngle")

SiStripLorentzAngleSim = cms.ESSource("PoolDBESSource",
                                      BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                      DBParameters = cms.PSet( messageLevel = cms.untracked.int32(0),
                                                               authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
                                                               ),
                                      timetype = cms.untracked.string('runnumber'),
                                      connect = cms.string(dbfile),
                                      toGet = cms.VPSet(
    cms.PSet( record = cms.string('SiStripLorentzAngleSimRcd'), tag = cms.string(dbtag) )
    )
                                      )
es_prefer_SiStripLorentzAngleSim = cms.ESPrefer("PoolDBESSource","SiStripLorentzAngleSim")

redigi_step       = cms.Path( pdigi )
