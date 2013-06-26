database = 'sqlite_file:fftjet_corr.db'
sequenceTag = "PF0"

import FWCore.ParameterSet.Config as cms 
from JetMETCorrections.FFTJetModules.fftjetcorrectionesproducer_cfi import *

process = cms.Process('FFTJetCorrectorDBRead') 

process.source = cms.Source('EmptySource') 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = database

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string(fftjet_corr_types[sequenceTag].dbRecord),
        tag = cms.string(fftjet_corr_types[sequenceTag].dbTag)
    ))
)

process.reader = cms.EDAnalyzer(
    'FFTJetCorrectorDBReader',
    record = cms.string(fftjet_corr_types[sequenceTag].dbRecord),
    outputFile = cms.string("fftjet_corr_restored.gssa"),
    printAsString = cms.bool(False),
    readArchive = cms.bool(True),
    isArchiveCompressed = cms.bool(False)
)

process.p = cms.Path(process.reader)
