database = "sqlite_file:fftjet_corr.db"
sequenceTag = "PF0"

import FWCore.ParameterSet.Config as cms
from JetMETCorrections.FFTJetModules.fftjetcorrectionesproducer_cfi import *

process = cms.Process('FFTJetCorrectorDBWrite') 

process.source = cms.Source('EmptySource') 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = database

# Data is tagged in the database by the "tag" parameter specified in the
# PoolDBOutputService configuration. We then check if the tag already exists.
#
# -- If the tag does not exist, a new interval of validity (IOV) for this tag
#    is created, valid till "end of time".
#
# -- If the tag already exists: the IOV of the previous data is stopped at
#    "current time" and we register new data valid from now on (currentTime
#    is the time of the current event!). 
#
# The "record" parameter should be the same in the PoolDBOutputService
# configuration and in the module which writes the object. It is basically
# used in order to just associate the record with the tag.
#
process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string(fftjet_corr_types[sequenceTag].dbRecord),
        tag = cms.string(fftjet_corr_types[sequenceTag].dbTag)
    ))
)

process.writer = cms.EDAnalyzer(
    'FFTJetCorrectorDBWriter',
    inputFile = cms.string("fftjet_corr.gssa"),
    record = cms.string(fftjet_corr_types[sequenceTag].dbRecord)
)

process.p = cms.Path(process.writer)
