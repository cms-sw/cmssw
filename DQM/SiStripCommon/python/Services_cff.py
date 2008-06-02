# The following comments couldn't be translated into the new config version:

# framework debugging...
# check memory
# subroutine-by-subroutine timing
# module-by-module timing
import FWCore.ParameterSet.Config as cms

Tracer = cms.Service("Tracer")

SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

SimpleProfiling = cms.Service("SimpleProfiling")

Timing = cms.Service("Timing")


