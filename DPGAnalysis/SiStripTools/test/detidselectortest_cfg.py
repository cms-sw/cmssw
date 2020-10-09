import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("Test")

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")

options.parseArguments()

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(186253),
                            numberEventsInRun = cms.untracked.uint32(1)
                            )

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring("detidselectorTest"),
                                    detidselectorTest = cms.untracked.PSet(
                                                    threshold = cms.untracked.string("DEBUG")
                                                    ),
                                    debugModules = cms.untracked.vstring("*")
                                    )


#process.detidselectortest = cms.EDAnalyzer("DetIdSelectorTest",
#                                           selections=cms.VPSet(
#    cms.PSet(selection=cms.untracked.vstring("0x1e0c0000-0x1c040000")),    # TEC minus
#    cms.PSet(selection=cms.untracked.vstring("0x1e0c0000-0x1c080000")),     # TEC plus
#    cms.PSet(selection=cms.untracked.vstring("0x1e000000-0x1a000000")),     # TOB
#    cms.PSet(selection=cms.untracked.vstring("0x1e000000-0x16000000")),     # TIB
#    cms.PSet(selection=cms.untracked.vstring("0x1e006000-0x18002000")),     # TID minus
#    cms.PSet(selection=cms.untracked.vstring("0x1e006000-0x18004000")),     # TID plus
#    cms.PSet(selection=cms.untracked.vstring("0x1e0f0000-0x12010000")),      # BPix L1
#    cms.PSet(selection=cms.untracked.vstring("0x1e0f0000-0x12020000")),      # BPix L2
#    cms.PSet(selection=cms.untracked.vstring("0x1e0f0000-0x12030000")),      # BPix L3
#    cms.PSet(selection=cms.untracked.vstring("0x1f800000-0x14800000")),      # FPix minus
#    cms.PSet(selection=cms.untracked.vstring("0x1f800000-0x15000000"))      # FPix plus
#    cms.PSet(selection=cms.untracked.vstring("504102912-470286336"))
#    )
#)

from DPGAnalysis.SiStripTools.occupancyplotsselections_simplified_cff import *

process.detidselectortest = cms.EDAnalyzer("DetIdSelectorTest",
                                           selections=cms.VPSet(
    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("TIDring1"),selection=cms.untracked.vstring("0x1e000600-0x18000200")),
    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("TIDring2"),selection=cms.untracked.vstring("0x1e000600-0x18000400")),
    cms.PSet(detSelection = cms.uint32(201),detLabel = cms.string("TECring1"),selection=cms.untracked.vstring("0x1e0000e0-0x1c000020")),
    cms.PSet(detSelection = cms.uint32(202),detLabel = cms.string("TECring2"),selection=cms.untracked.vstring("0x1e0000e0-0x1c000040"))
    )
)
#process.detidselectortest.selections.extend(OccupancyPlotsStripWantedSubDets)
#process.detidselectortest.selections.extend(OccupancyPlotsPixelWantedSubDets)

process.load("DQM.SiStripCommon.TkHistoMap_cff")

#process.Timing = cms.Service("Timing")


# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))

process.p = cms.Path(process.detidselectortest)



