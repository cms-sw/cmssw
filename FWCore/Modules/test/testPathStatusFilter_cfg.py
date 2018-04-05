import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource")

# This is the expression in ModuloEventIDFilter "iEvent.id().event() % n_ == offset_"
process.allPass = cms.EDFilter("ModuloEventIDFilter",
    modulo = cms.uint32(1),
    offset = cms.uint32(0)
)

process.allFail = cms.EDFilter("ModuloEventIDFilter",
    modulo = cms.uint32(1),
    offset = cms.uint32(0)
)
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

process.pathAllPass = cms.Path(process.allPass)
process.pathAllFail = cms.Path(process.allFail)
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

# test 'and'
process.pathStatusFilter2 = cms.EDFilter('PathStatusFilter',
  logicalExpression = cms.string('pathMod2 and pathMod5'),
  verbose = cms.untracked.bool(False)
)
process.testpath2 = cms.Path(process.pathStatusFilter2)

process.sewer2 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(2),
    name = cms.string('for_testpath2'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('testpath2')
    )
)

# test 'or'
process.pathStatusFilter3 = cms.EDFilter('PathStatusFilter',
  logicalExpression = cms.string('pathMod2 or pathMod5'),
  verbose = cms.untracked.bool(False)
)
process.testpath3 = cms.Path(process.pathStatusFilter3)

process.sewer3 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(12),
    name = cms.string('for_testpath3'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('testpath3')
    )
)

# test 'not'
process.pathStatusFilter4 = cms.EDFilter('PathStatusFilter',
  logicalExpression = cms.string('not pathMod5'),
  verbose = cms.untracked.bool(False)
)
process.testpath4 = cms.Path(process.pathStatusFilter4)

process.sewer4 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(16),
    name = cms.string('for_testpath4'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('testpath4')
    )
)

# test precedence
process.pathStatusFilter5 = cms.EDFilter('PathStatusFilter',
  logicalExpression = cms.string('pathMod2 or pathMod3 and pathMod5'),
  verbose = cms.untracked.bool(False)
)
process.testpath5 = cms.Path(process.pathStatusFilter5)

process.sewer5 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(11),
    name = cms.string('for_testpath5'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('testpath5')
    )
)

# test parentheses
process.pathStatusFilter6 = cms.EDFilter('PathStatusFilter',
  logicalExpression = cms.string('(pathMod2 or pathMod3) and pathMod5'),
  verbose = cms.untracked.bool(False)
)
process.testpath6 = cms.Path(process.pathStatusFilter6)

process.sewer6 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(3),
    name = cms.string('for_testpath6'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('testpath6')
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

# test empty string for expression
process.pathStatusFilter8 = cms.EDFilter('PathStatusFilter',
  logicalExpression = cms.string(""),
  verbose = cms.untracked.bool(False)
)
process.testpath8 = cms.Path(process.pathStatusFilter8)

process.sewer8 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(20),
    name = cms.string('for_testpath8'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('testpath8')
    )
)

# test single character pathname
process.a = cms.Path(process.allPass)
process.pathStatusFilter9 = cms.EDFilter('PathStatusFilter',
  logicalExpression = cms.string("a"),
  verbose = cms.untracked.bool(False)
)
process.testpath9 = cms.Path(process.pathStatusFilter9)

process.sewer9 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(20),
    name = cms.string('for_testpath9'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('testpath9')
    )
)

# test pathnames containing an operator name and duplicate pathnames
process.nota = cms.Path(process.allPass)
process.anot = cms.Path(process.allPass)
process.pathStatusFilter10 = cms.EDFilter('PathStatusFilter',
  logicalExpression = cms.string("nota and nota and anot"),
  verbose = cms.untracked.bool(False)
)
process.testpath10 = cms.Path(process.pathStatusFilter10)

process.sewer10 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(20),
    name = cms.string('for_testpath10'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('testpath10')
    )
)

# test multiple parentheses and 'not's
process.pathStatusFilter11 = cms.EDFilter('PathStatusFilter',
  logicalExpression = cms.string('not not not (((not(not(((((not not pathMod2))) or pathMod3))) and pathMod5)))'),
  verbose = cms.untracked.bool(False)
)
process.testpath11 = cms.Path(process.pathStatusFilter11)

process.sewer11 = cms.OutputModule("SewerModule",
    shouldPass = cms.int32(17),
    name = cms.string('for_testpath11'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('testpath11')
    )
)

process.endpath = cms.EndPath(process.sewer1 *
                              process.sewer2 *
                              process.sewer3 *
                              process.sewer4 *
                              process.sewer5 *
                              process.sewer6 *
                              process.sewer7 *
                              process.sewer8 *
                              process.sewer9 *
                              process.sewer10 *
                              process.sewer11
)
