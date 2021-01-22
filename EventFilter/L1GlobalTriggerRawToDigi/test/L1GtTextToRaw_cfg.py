#
# cfg file for a module to produce a raw GT DAQ or EVM record 
# starting from a text (ASCII) file
#

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("TestGtTextToRaw")

# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# source file: ASCII file, hex format, one 64bits word per line
# some standard dumps could be given directly and will be cleaned  
process.source = cms.Source("EmptySource")

# load and configure modules

# L1 EventSetup
process.load("L1Trigger.Configuration.L1DummyConfig_cff")

# L1GtTextToRaw
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtTextToRaw_cfi")

# text file type: indicate the producer of the file
# for standard dumps, the module will clean the file
#process.l1GtTextToRaw.TextFileType = 'VmeSpyDump'

process.l1GtTextToRaw.TextFileName = '/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_TextToRaw_source.txt'

# FED Id (default 813)
#process.l1GtTextToRaw.DaqGtFedId = 813
    
# FED raw data size (in 8bits units, including header and trailer)
# If negative value, the size is retrieved from the trailer.        
#process.l1GtTextToRaw.RawDataSize = 872

# path to be run
process.p = cms.Path(process.l1GtTextToRaw)

# services

# Message Logger
# uncomment / comment messages with DEBUG mode to run in DEBUG mode
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('l1GtTextToRaw'),
    files = cms.untracked.PSet(
        testGt_TextToRaw = cms.untracked.PSet(
            DEBUG = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            INFO = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            threshold = cms.untracked.string('DEBUG')
        )
    )
)

# output 
process.outputL1GtTextToRaw = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testGt_TextToRaw_output.root'),
    # keep only packed data in the ROOT file
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_l1GtTextToRaw_*_*')
)

process.outpath = cms.EndPath(process.outputL1GtTextToRaw)

