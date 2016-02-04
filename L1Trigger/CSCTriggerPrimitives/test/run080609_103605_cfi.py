import FWCore.ParameterSet.Config as cms

source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string("CSCFileReader"),
    readerPset = cms.untracked.PSet(
        firstEvent = cms.untracked.int32(0),
        RUI00 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI00_Default_000_080609_103605_UTC.raw"),
        RUI01 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI01_Default_000_080609_103605_UTC.raw"),
        RUI02 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI02_Default_000_080609_103605_UTC.raw"),
        RUI03 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI03_Default_000_080609_103605_UTC.raw"),
        RUI04 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI04_Default_000_080609_103605_UTC.raw"),
        RUI05 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI05_Default_000_080609_103605_UTC.raw"),
        RUI06 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI06_Default_000_080609_103605_UTC.raw"),
        RUI07 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI07_Default_000_080609_103605_UTC.raw"),
        RUI08 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI08_Default_000_080609_103605_UTC.raw"),
        RUI09 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI09_Default_000_080609_103605_UTC.raw"),
        RUI10 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI10_Default_000_080609_103605_UTC.raw"),
        RUI11 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI11_Default_000_080609_103605_UTC.raw"),
        RUI12 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI12_Default_000_080609_103605_UTC.raw"),
        RUI13 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI13_Default_000_080609_103605_UTC.raw"),
        RUI14 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI14_Default_000_080609_103605_UTC.raw"),
        RUI15 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI15_Default_000_080609_103605_UTC.raw"),
        RUI16 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI16_Default_000_080609_103605_UTC.raw"),
        RUI17 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI17_Default_000_080609_103605_UTC.raw"),
        RUI18 = cms.untracked.vstring("/data0/slava/data/run080609_103605/csc_00000000_EmuRUI18_Default_000_080609_103605_UTC.raw"),
        FED750 = cms.untracked.vstring("RUI01", "RUI02", "RUI03", "RUI04", "RUI05", "RUI06", "RUI07", "RUI08", "RUI09", "RUI10"),
        FED751 = cms.untracked.vstring("RUI11", "RUI12", "RUI13", "RUI14", "RUI15", "RUI16", "RUI17", "RUI18"),
        FED760 = cms.untracked.vstring("RUI00")
    )
)
