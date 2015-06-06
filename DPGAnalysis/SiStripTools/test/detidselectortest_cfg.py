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

process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
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
    cms.PSet(detLabel = cms.string("noisy"),selection=cms.untracked.vstring("0x1fffffff-0x1a00a4f1"))
#    cms.PSet(detLabel = cms.string("TECplus_3_8_4_2_3_ring5"),selection=cms.untracked.vstring("0x1fffffe0-0x1c0a13a0")),
#    cms.PSet(detLabel = cms.string("TECplus_3_8_4_2_3_ring7"),selection=cms.untracked.vstring("0x1fffffe0-0x1c0a13e0")),
#    cms.PSet(detLabel = cms.string("TECminus_5_4_4_2_3_ring5"),selection=cms.untracked.vstring("0x1fffffe0-0x1c0515a0")),
#    cms.PSet(detLabel = cms.string("TECminus_5_4_4_2_3_ring7"),selection=cms.untracked.vstring("0x1fffffe0-0x1c0515e0"))
    )
)
#process.detidselectortest.selections.extend(OccupancyPlotsStripWantedSubDets)
#process.detidselectortest.selections.extend(OccupancyPlotsPixelWantedSubDets)

process.DQMStore = cms.Service("DQMStore")
process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

#process.Timing = cms.Service("Timing")


# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))

process.p = cms.Path(process.detidselectortest)



