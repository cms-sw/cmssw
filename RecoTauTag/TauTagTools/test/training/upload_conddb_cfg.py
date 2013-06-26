#!/usr/bin/env cmsRun

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import sys

options = VarParsing.VarParsing ('standard')

options.register ('sourcetag',
                  'Tanc', # default value
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Name of tag in local database")

options.register ('totag',
                  '', # default value
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Name of tag in destination database")

options.register ('source',
                  'sqlite_file:TancLocal.db', # default value
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Database to copy payload from")

options.register ('to',
                  #'oracle://cms_orcoff_prep/CMS_COND_31X_BTAU', # default value
                  'oracle://cms_orcoff_prep/CMS_COND_31X_BTAU', # default value
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Database to copy payload to")

options.parseArguments()

print "***************************************************"
print "******  Upload Tau Conditions to DB          ******"
print "***************************************************"
print "*  DB in:        %s                               " % options.source
print "*  tag in:       %s                               " % options.sourcetag
print "*  DB out:       %s                               " % options.to
print "*  tag out:      %s                               " % options.totag
print "* ----------------------------------------------- "

if not options.totag:
    print "You must specify an output tag! [totag]"
    sys.exit(1)

process = cms.Process("DBupload")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

# Setup from database
process.load("RecoTauTag.TauTagTools.TancConditions_cff")
process.TauTagMVAComputerRecord.connect = cms.string(options.source)
process.TauTagMVAComputerRecord.toGet[0].tag = options.sourcetag

process.saver = cms.EDAnalyzer(
    "TauMVATrainerSave",
    toPut = cms.vstring(),
    toCopy = cms.vstring(
        '1prong0pi0',
        '1prong1pi0',
        '1prong2pi0',
        '3prong0pi0',
    )
)

# Setup output database
process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(5),
        #authenticationPath = cms.untracked.string(
        #    '/afs/cern.ch/cms/DB/conddb/readWritePrep.xml'),
        #authenticationPath = cms.untracked.string('.'),
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string(options.to),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('TauTagMVAComputerRcd'),
        tag = cms.string(options.totag)
    ))
)

process.outputpath = cms.EndPath(process.saver)
