import FWCore.ParameterSet.Config as cms

TotemRawToDigi = cms.EDProducer("TotemRawToDigi",
  rawDataTag = cms.InputTag(""),

  RawUnpacking = cms.PSet(
  ),

  RawToDigi = cms.PSet(
    # 0: no error output
    # 1: one-line message for every event with at least one corrupted VFAT frame
    # 2: lists all corrupted VFATs in all events
    # 3: lists all corruptions for all corrupted VFATs in all events
    verbosity = cms.untracked.uint32(0),

    # flags for available consistency tests
    # 0: do not perform the test at all
    # 1: print an error message, but keep the frame
    # 2: print an error message and do not process the frame
    testFootprint = cms.uint32(2),
    testCRC = cms.uint32(2),
    testID = cms.uint32(2),               # compare the ID from data and mapping
    testECMostFrequent = cms.uint32(2),   # compare frame's EC with the most frequent value in the event
    testBCMostFrequent = cms.uint32(2),   # compare frame's BC with the most frequent value in the event
    
    # if non-zero, prints a per-VFAT error summary at the end of the job
    printErrorSummary = cms.untracked.uint32(1),
    
    # if non-zero, prints a summary of frames found in data, but not in the mapping
    printUnknownFrameSummary = cms.untracked.uint32(0),

    # output labels
    rpCCProductLabel = cms.untracked.string("rpCCOutput"),
    rpDataProductLabel = cms.untracked.string("rpDataOutput"),
    t1DataProductLabel = cms.untracked.string("t1DataOutput"),
    t2DataProductLabel = cms.untracked.string("t2DataOutput"),
    conversionStatusLabel = cms.untracked.string("conversionStatus"),
    
    # the minimal number of frames to search for the most frequent counter value 
    EC_min = cms.untracked.uint32(10),
    BC_min = cms.untracked.uint32(10),

    # the most frequent counter value is accepted provided its relative occupancy is higher than this fraction
    EC_fraction = cms.untracked.double(0.6),
    BC_fraction = cms.untracked.double(0.6),
  )
)
