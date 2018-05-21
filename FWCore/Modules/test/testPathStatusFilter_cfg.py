# Tests the PathStatusFilter module
# Note many of the tests that were originally here
# were moved to test_catch2_PathStatusFilter.cc.

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource")

# This is the expression in ModuloEventIDFilter "iEvent.id().event() % n_ == offset_"

# pass 2, 4, 6, 8 ...
process.mod2 = cms.EDFilter("ModuloEventIDFilter",
    modulo = cms.uint32(2),
    offset = cms.uint32(0)
)
process.passOdd = cms.EDFilter("ModuloEventIDFilter",
    modulo = cms.uint32(2),
    offset = cms.uint32(1)
)
process.mod3 = cms.EDFilter("ModuloEventIDFilter",
    modulo = cms.uint32(3),
    offset = cms.uint32(0)
)
process.mod5 = cms.EDFilter("ModuloEventIDFilter",
    modulo = cms.uint32(5),
    offset = cms.uint32(0)
)
process.mod7 = cms.EDFilter("ModuloEventIDFilter",
    modulo = cms.uint32(7),
    offset = cms.uint32(0)
)
process.mod15 = cms.EDFilter("ModuloEventIDFilter",
    modulo = cms.uint32(15),
    offset = cms.uint32(0)
)
process.mod20 = cms.EDFilter("ModuloEventIDFilter",
    modulo = cms.uint32(20),
    offset = cms.uint32(0)
)

process.pathPassOdd = cms.Path(process.passOdd)

process.pathMod2 = cms.Path(process.mod2)
process.pathMod3 = cms.Path(process.mod3)
process.pathMod5 = cms.Path(process.mod5)
process.pathMod7 = cms.Path(process.mod7)
process.pathMod15 = cms.Path(process.mod15)
process.pathMod20 = cms.Path(process.mod20)

# test pathname with all allowed characters types
process.pathMod2_and_Mod3 = cms.Path(process.mod2 * process.mod3)

process.load("FWCore.Modules.pathStatusFilter_cfi")
process.pathStatusFilter.logicalExpression = 'pathMod2_and_Mod3'
# Set True for debugging parsing of logical expressions and
# their conversion into trees of operators and operands
process.pathStatusFilter.verbose = False

process.testpath1 = cms.Path(process.pathStatusFilter)

process.sewer1 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(3),
    name = cms.string('for_testpath1'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('testpath1')
    )
)

# test extra space between pathnames and operators
process.pathStatusFilter7 = cms.EDFilter('PathStatusFilter',
  logicalExpression = cms.string("\t    pathMod2   \t and \t  pathMod3 \t and \t not(pathMod5)and not not not pathMod7 or(pathMod15)and  not (  pathMod20 or(pathPassOdd)) \t   "),
  verbose = cms.untracked.bool(False)
)
process.testpath7 = cms.Path(process.pathStatusFilter7)

process.sewer7 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(3),
    name = cms.string('for_testpath7'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('testpath7')
    )
)

process.endpath = cms.EndPath(process.sewer1 * process.sewer7)
