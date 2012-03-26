import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("Test")

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST::All",
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
process.load("Configuration.StandardSequences.Geometry_cff")
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
##    cms.PSet(selection=cms.untracked.vstring("504102912-470286336"))
#    )
#)

from DPGAnalysis.SiStripTools.occupancyplotsselections_simplified_cff import *

process.detidselectortest = cms.EDAnalyzer("DetIdSelectorTest",
                                           selections=cms.VPSet(
    cms.PSet(detSelection=cms.uint32(4110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c044020")),    # TEC- D1 R1 back
    cms.PSet(detSelection=cms.uint32(4120),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c048020")),    # TEC- D2 R1 back
    cms.PSet(detSelection=cms.uint32(4130),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c020")),    # TEC- D3 R1 back
#    cms.PSet(detSelection=cms.uint32(4140),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c050020")),    # TEC- D4 R1 back
#    cms.PSet(detSelection=cms.uint32(4150),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c054020")),    # TEC- D5 R1 back
#    cms.PSet(detSelection=cms.uint32(4160),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c058020")),    # TEC- D6 R1 back
#    cms.PSet(detSelection=cms.uint32(4170),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c020")),    # TEC- D7 R1 back
#    cms.PSet(detSelection=cms.uint32(4180),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c060020")),    # TEC- D8 R1 back
#    cms.PSet(detSelection=cms.uint32(4190),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c064020")),    # TEC- D9 R1 back

    cms.PSet(detSelection=cms.uint32(4210),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c044040")),    # TEC- D1 R2 back
    cms.PSet(detSelection=cms.uint32(4220),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c048040")),    # TEC- D2 R2 back
    cms.PSet(detSelection=cms.uint32(4230),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c040")),    # TEC- D3 R2 back
    cms.PSet(detSelection=cms.uint32(4240),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c050040")),    # TEC- D4 R2 back
    cms.PSet(detSelection=cms.uint32(4250),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c054040")),    # TEC- D5 R2 back
    cms.PSet(detSelection=cms.uint32(4260),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c058040")),    # TEC- D6 R2 back
#    cms.PSet(detSelection=cms.uint32(4270),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c040")),    # TEC- D7 R2 back
#    cms.PSet(detSelection=cms.uint32(4280),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c060040")),    # TEC- D8 R2 back
#    cms.PSet(detSelection=cms.uint32(4290),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c064040")),    # TEC- D9 R2 back

    cms.PSet(detSelection=cms.uint32(4310),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c044060")),    # TEC- D1 R3 back
    cms.PSet(detSelection=cms.uint32(4320),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c048060")),    # TEC- D2 R3 back
    cms.PSet(detSelection=cms.uint32(4330),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c060")),    # TEC- D3 R3 back
    cms.PSet(detSelection=cms.uint32(4340),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c050060")),    # TEC- D4 R3 back
    cms.PSet(detSelection=cms.uint32(4350),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c054060")),    # TEC- D5 R3 back
    cms.PSet(detSelection=cms.uint32(4360),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c058060")),    # TEC- D6 R3 back
    cms.PSet(detSelection=cms.uint32(4370),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c060")),    # TEC- D7 R3 back
    cms.PSet(detSelection=cms.uint32(4380),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c060060")),    # TEC- D8 R3 back
#    cms.PSet(detSelection=cms.uint32(4390),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c064060")),    # TEC- D9 R3 back

    cms.PSet(detSelection=cms.uint32(4410),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c044080")),    # TEC- D1 R4 back
    cms.PSet(detSelection=cms.uint32(4420),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c048080")),    # TEC- D2 R4 back
    cms.PSet(detSelection=cms.uint32(4430),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c080")),    # TEC- D3 R4 back
    cms.PSet(detSelection=cms.uint32(4440),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c050080")),    # TEC- D4 R4 back
    cms.PSet(detSelection=cms.uint32(4450),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c054080")),    # TEC- D5 R4 back
    cms.PSet(detSelection=cms.uint32(4460),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c058080")),    # TEC- D6 R4 back
    cms.PSet(detSelection=cms.uint32(4470),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c080")),    # TEC- D7 R4 back
    cms.PSet(detSelection=cms.uint32(4480),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c060080")),    # TEC- D8 R4 back
    cms.PSet(detSelection=cms.uint32(4490),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c064080")),    # TEC- D9 R4 back
    
    cms.PSet(detSelection=cms.uint32(4510),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0440a0")),    # TEC- D1 R5 back
    cms.PSet(detSelection=cms.uint32(4520),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0480a0")),    # TEC- D2 R5 back
    cms.PSet(detSelection=cms.uint32(4530),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c0a0")),    # TEC- D3 R5 back
    cms.PSet(detSelection=cms.uint32(4540),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0500a0")),    # TEC- D4 R5 back
    cms.PSet(detSelection=cms.uint32(4550),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0540a0")),    # TEC- D5 R5 back
    cms.PSet(detSelection=cms.uint32(4560),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0580a0")),    # TEC- D6 R5 back
    cms.PSet(detSelection=cms.uint32(4570),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c0a0")),    # TEC- D7 R5 back
    cms.PSet(detSelection=cms.uint32(4580),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0600a0")),    # TEC- D8 R5 back
    cms.PSet(detSelection=cms.uint32(4590),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0640a0")),    # TEC- D9 R5 back

    cms.PSet(detSelection=cms.uint32(4610),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0440c0")),    # TEC- D1 R6 back
    cms.PSet(detSelection=cms.uint32(4620),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0480c0")),    # TEC- D2 R6 back
    cms.PSet(detSelection=cms.uint32(4630),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c0c0")),    # TEC- D3 R6 back
    cms.PSet(detSelection=cms.uint32(4640),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0500c0")),    # TEC- D4 R6 back
    cms.PSet(detSelection=cms.uint32(4650),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0540c0")),    # TEC- D5 R6 back
    cms.PSet(detSelection=cms.uint32(4660),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0580c0")),    # TEC- D6 R6 back
    cms.PSet(detSelection=cms.uint32(4670),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c0c0")),    # TEC- D7 R6 back
    cms.PSet(detSelection=cms.uint32(4680),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0600c0")),    # TEC- D8 R6 back
    cms.PSet(detSelection=cms.uint32(4690),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0640c0")),    # TEC- D9 R6 back

    cms.PSet(detSelection=cms.uint32(4710),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0440e0")),    # TEC- D1 R7 back
    cms.PSet(detSelection=cms.uint32(4720),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0480e0")),    # TEC- D2 R7 back
    cms.PSet(detSelection=cms.uint32(4730),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c04c0e0")),    # TEC- D3 R7 back
    cms.PSet(detSelection=cms.uint32(4740),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0500e0")),    # TEC- D4 R7 back
    cms.PSet(detSelection=cms.uint32(4750),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0540e0")),    # TEC- D5 R7 back
    cms.PSet(detSelection=cms.uint32(4760),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0580e0")),    # TEC- D6 R7 back
    cms.PSet(detSelection=cms.uint32(4770),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c05c0e0")),    # TEC- D7 R7 back
    cms.PSet(detSelection=cms.uint32(4780),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0600e0")),    # TEC- D8 R7 back
    cms.PSet(detSelection=cms.uint32(4790),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0640e0")),    # TEC- D9 R7 back



    cms.PSet(detSelection=cms.uint32(5110),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c084020")),    # TEC+ D1 R1 back
    cms.PSet(detSelection=cms.uint32(5120),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c088020")),    # TEC+ D2 R1 back
    cms.PSet(detSelection=cms.uint32(5130),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c020")),    # TEC+ D3 R1 back
#    cms.PSet(detSelection=cms.uint32(5140),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c090020")),    # TEC+ D4 R1 back
#    cms.PSet(detSelection=cms.uint32(5150),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c094020")),    # TEC+ D5 R1 back
#    cms.PSet(detSelection=cms.uint32(5160),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c098020")),    # TEC+ D6 R1 back
#    cms.PSet(detSelection=cms.uint32(5170),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c020")),    # TEC+ D7 R1 back
#    cms.PSet(detSelection=cms.uint32(5180),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a0020")),    # TEC+ D8 R1 back
#    cms.PSet(detSelection=cms.uint32(5190),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a4020")),    # TEC+ D9 R1 back


    cms.PSet(detSelection=cms.uint32(5210),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c084040")),    # TEC+ D1 R2 back
    cms.PSet(detSelection=cms.uint32(5220),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c088040")),    # TEC+ D2 R2 back
    cms.PSet(detSelection=cms.uint32(5230),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c040")),    # TEC+ D3 R2 back
    cms.PSet(detSelection=cms.uint32(5240),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c090040")),    # TEC+ D4 R2 back
    cms.PSet(detSelection=cms.uint32(5250),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c094040")),    # TEC+ D5 R2 back
    cms.PSet(detSelection=cms.uint32(5260),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c098040")),    # TEC+ D6 R2 back
#    cms.PSet(detSelection=cms.uint32(5270),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c040")),    # TEC+ D7 R2 back
#    cms.PSet(detSelection=cms.uint32(5280),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a0040")),    # TEC+ D8 R2 back
#    cms.PSet(detSelection=cms.uint32(5290),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a4040")),    # TEC+ D9 R2 back

    cms.PSet(detSelection=cms.uint32(5310),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c084060")),    # TEC+ D1 R3 back
    cms.PSet(detSelection=cms.uint32(5320),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c088060")),    # TEC+ D2 R3 back
    cms.PSet(detSelection=cms.uint32(5330),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c060")),    # TEC+ D3 R3 back
    cms.PSet(detSelection=cms.uint32(5340),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c090060")),    # TEC+ D4 R3 back
    cms.PSet(detSelection=cms.uint32(5350),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c094060")),    # TEC+ D5 R3 back
    cms.PSet(detSelection=cms.uint32(5360),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c098060")),    # TEC+ D6 R3 back
    cms.PSet(detSelection=cms.uint32(5370),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c060")),    # TEC+ D7 R3 back
    cms.PSet(detSelection=cms.uint32(5380),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a0060")),    # TEC+ D8 R3 back
#    cms.PSet(detSelection=cms.uint32(5390),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a4060")),    # TEC+ D9 R3 back

    cms.PSet(detSelection=cms.uint32(5410),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c084080")),    # TEC+ D1 R4 back
    cms.PSet(detSelection=cms.uint32(5420),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c088080")),    # TEC+ D2 R4 back
    cms.PSet(detSelection=cms.uint32(5430),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c080")),    # TEC+ D3 R4 back
    cms.PSet(detSelection=cms.uint32(5440),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c090080")),    # TEC+ D4 R4 back
    cms.PSet(detSelection=cms.uint32(5450),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c094080")),    # TEC+ D5 R4 back
    cms.PSet(detSelection=cms.uint32(5460),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c098080")),    # TEC+ D6 R4 back
    cms.PSet(detSelection=cms.uint32(5470),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c080")),    # TEC+ D7 R4 back
    cms.PSet(detSelection=cms.uint32(5480),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a0080")),    # TEC+ D8 R4 back
    cms.PSet(detSelection=cms.uint32(5490),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a4080")),    # TEC+ D9 R4 back

    cms.PSet(detSelection=cms.uint32(5510),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0840a0")),    # TEC+ D1 R5 back
    cms.PSet(detSelection=cms.uint32(5520),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0880a0")),    # TEC+ D2 R5 back
    cms.PSet(detSelection=cms.uint32(5530),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c0a0")),    # TEC+ D3 R5 back
    cms.PSet(detSelection=cms.uint32(5540),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0900a0")),    # TEC+ D4 R5 back
    cms.PSet(detSelection=cms.uint32(5550),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0940a0")),    # TEC+ D5 R5 back
    cms.PSet(detSelection=cms.uint32(5560),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0980a0")),    # TEC+ D6 R5 back
    cms.PSet(detSelection=cms.uint32(5570),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c0a0")),    # TEC+ D7 R5 back
    cms.PSet(detSelection=cms.uint32(5580),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a00a0")),    # TEC+ D8 R5 back
    cms.PSet(detSelection=cms.uint32(5590),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a40a0")),    # TEC+ D9 R5 back

    cms.PSet(detSelection=cms.uint32(5610),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0840c0")),    # TEC+ D1 R6 back
    cms.PSet(detSelection=cms.uint32(5620),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0880c0")),    # TEC+ D2 R6 back
    cms.PSet(detSelection=cms.uint32(5630),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c0c0")),    # TEC+ D3 R6 back
    cms.PSet(detSelection=cms.uint32(5640),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0900c0")),    # TEC+ D4 R6 back
    cms.PSet(detSelection=cms.uint32(5650),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0940c0")),    # TEC+ D5 R6 back
    cms.PSet(detSelection=cms.uint32(5660),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0980c0")),    # TEC+ D6 R6 back
    cms.PSet(detSelection=cms.uint32(5670),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c0c0")),    # TEC+ D7 R6 back
    cms.PSet(detSelection=cms.uint32(5680),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a00c0")),    # TEC+ D8 R6 back
    cms.PSet(detSelection=cms.uint32(5690),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a40c0")),    # TEC+ D9 R6 back

    cms.PSet(detSelection=cms.uint32(5710),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0840e0")),    # TEC+ D1 R7 back
    cms.PSet(detSelection=cms.uint32(5720),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0880e0")),    # TEC+ D2 R7 back
    cms.PSet(detSelection=cms.uint32(5730),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c08c0e0")),    # TEC+ D3 R7 back
    cms.PSet(detSelection=cms.uint32(5740),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0900e0")),    # TEC+ D4 R7 back
    cms.PSet(detSelection=cms.uint32(5750),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0940e0")),    # TEC+ D5 R7 back
    cms.PSet(detSelection=cms.uint32(5760),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0980e0")),    # TEC+ D6 R7 back
    cms.PSet(detSelection=cms.uint32(5770),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c09c0e0")),    # TEC+ D7 R7 back
    cms.PSet(detSelection=cms.uint32(5780),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a00e0")),    # TEC+ D8 R7 back
    cms.PSet(detSelection=cms.uint32(5790),detLabel=cms.string("Dummy"),selection=cms.untracked.vstring("0x1e0fc0e0-0x1c0a40e0"))    # TEC+ D9 R7 back
    )
)
#process.detidselectortest.selections.extend(OccupancyPlotsStripWantedSubDets)
#process.detidselectortest.selections.extend(OccupancyPlotsPixelWantedSubDets)

process.DQMStore = cms.Service("DQMStore")
process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

process.Timing = cms.Service("Timing")


# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = options.globalTag

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))

process.p = cms.Path(process.detidselectortest)



