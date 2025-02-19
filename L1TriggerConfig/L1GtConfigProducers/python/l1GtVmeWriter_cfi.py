import FWCore.ParameterSet.Config as cms

# cfi for L1 GT VME Writer
l1GtVmeWriter = cms.EDFilter("L1GtVmeWriter",
                             
    # choose ouput directory
    OutputDir = cms.string('../test/output/')
)


