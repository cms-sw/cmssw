import FWCore.ParameterSet.Config as cms

totemVFATRawToDigi = cms.EDProducer("TotemVFATRawToDigi",
  rawDataTag = cms.InputTag(""),

  # options: "RP"
  subSystem = cms.string(""),

  # IMPORTANT: leave empty to load the default configuration from
  #    DataFormats/FEDRawData/interface/FEDNumbering.h
  fedIds = cms.vuint32(),

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

    # the minimal number of frames to search for the most frequent counter value 
    EC_min = cms.untracked.uint32(10),
    BC_min = cms.untracked.uint32(10),

    # the most frequent counter value is accepted provided its relative occupancy is higher than this fraction
    EC_fraction = cms.untracked.double(0.6),
    BC_fraction = cms.untracked.double(0.6),
    
    # if non-zero, prints a per-VFAT error summary at the end of the job
    printErrorSummary = cms.untracked.uint32(1),
    
    # if non-zero, prints a summary of frames found in data, but not in the mapping
    printUnknownFrameSummary = cms.untracked.uint32(0),
  )
)
