#!/usr/bin/env cmsRun

import FWCore.ParameterSet.Config as cms
import sys
import os

process = cms.Process("merge_dbs")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(	input = cms.untracked.int32(1) )

output_file = sys.argv[2]
input_files = sys.argv[3:]

print "Merging input files: %s into %s" % (
    " ".join(input_files), output_file)

def get_mva_name(file):
    # map db/1prong1pi0_blah.mva -> 1prong1pi0
    filename = os.path.splitext(os.path.basename(file))[0]
    return filename.split('_')[0]

input_config = dict((get_mva_name(file), cms.string(file))
                    for file in input_files)

process.db_source = cms.ESSource(
    "TauMVAComputerESSource",
    **input_config
)

process.saver = cms.EDAnalyzer(
    "TauMVATrainerSave",
    toCopy = cms.vstring(input_config.keys()),
    toPut = cms.vstring()
)

process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet( messageLevel = cms.untracked.int32(4) ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite:%s' % output_file),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('TauTagMVAComputerRcd'),
        tag = cms.string("Tanc")
    ))
)

process.outpath = cms.EndPath(process.saver)
