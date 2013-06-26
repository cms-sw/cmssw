import FWCore.ParameterSet.Config as cms

# run = 59318
# run = 62232 # reference run ("splash" events)
# run = 65882
# run = 67926 # RUI00-18
# run = 67912 # RUI00-18
# run = 72038
# run = 78396 # no HV
# run = 79652 # new TMB firmware in one crate
run = 80733 # new TMB firmware in all crates except ME-1
dir =  "/data0/slava/data/run" + str(run) + "/"
file = "csc_000" + str(run) + "_"
ext = "_Monitor_000.raw"
# print dir+file+"EmuRUI00"+ext

source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string("CSCFileReader"),
    readerPset = cms.untracked.PSet(
        #firstEvent = cms.untracked.int32(62550),
        #RUI00 = cms.untracked.vstring(dir+file+"EmuRUI00"+ext),
        RUI01 = cms.untracked.vstring(dir+file+"EmuRUI01"+ext),
        RUI02 = cms.untracked.vstring(dir+file+"EmuRUI02"+ext),
        RUI03 = cms.untracked.vstring(dir+file+"EmuRUI03"+ext),
        RUI04 = cms.untracked.vstring(dir+file+"EmuRUI04"+ext),
        RUI05 = cms.untracked.vstring(dir+file+"EmuRUI05"+ext),
        RUI06 = cms.untracked.vstring(dir+file+"EmuRUI06"+ext),
        RUI07 = cms.untracked.vstring(dir+file+"EmuRUI07"+ext),
        RUI08 = cms.untracked.vstring(dir+file+"EmuRUI08"+ext),
        RUI09 = cms.untracked.vstring(dir+file+"EmuRUI09"+ext),
        RUI10 = cms.untracked.vstring(dir+file+"EmuRUI10"+ext),
        RUI11 = cms.untracked.vstring(dir+file+"EmuRUI11"+ext),
        RUI12 = cms.untracked.vstring(dir+file+"EmuRUI12"+ext),
        RUI13 = cms.untracked.vstring(dir+file+"EmuRUI13"+ext),
        RUI14 = cms.untracked.vstring(dir+file+"EmuRUI14"+ext),
        RUI15 = cms.untracked.vstring(dir+file+"EmuRUI15"+ext),
        RUI16 = cms.untracked.vstring(dir+file+"EmuRUI16"+ext),
        RUI17 = cms.untracked.vstring(dir+file+"EmuRUI17"+ext),
        RUI18 = cms.untracked.vstring(dir+file+"EmuRUI18"+ext),
        RUI19 = cms.untracked.vstring(dir+file+"EmuRUI19"+ext),
        RUI20 = cms.untracked.vstring(dir+file+"EmuRUI20"+ext),
        RUI21 = cms.untracked.vstring(dir+file+"EmuRUI21"+ext),
        RUI22 = cms.untracked.vstring(dir+file+"EmuRUI22"+ext),
        RUI23 = cms.untracked.vstring(dir+file+"EmuRUI23"+ext),
        RUI24 = cms.untracked.vstring(dir+file+"EmuRUI24"+ext),
        RUI25 = cms.untracked.vstring(dir+file+"EmuRUI25"+ext),
        RUI26 = cms.untracked.vstring(dir+file+"EmuRUI26"+ext),
        RUI27 = cms.untracked.vstring(dir+file+"EmuRUI27"+ext),
        RUI28 = cms.untracked.vstring(dir+file+"EmuRUI28"+ext),
        RUI29 = cms.untracked.vstring(dir+file+"EmuRUI29"+ext),
        RUI30 = cms.untracked.vstring(dir+file+"EmuRUI30"+ext),
        RUI31 = cms.untracked.vstring(dir+file+"EmuRUI31"+ext),
        RUI32 = cms.untracked.vstring(dir+file+"EmuRUI32"+ext),
        RUI33 = cms.untracked.vstring(dir+file+"EmuRUI33"+ext),
        RUI34 = cms.untracked.vstring(dir+file+"EmuRUI34"+ext),
        RUI35 = cms.untracked.vstring(dir+file+"EmuRUI35"+ext),
        RUI36 = cms.untracked.vstring(dir+file+"EmuRUI36"+ext),
        FED750 = cms.untracked.vstring("RUI01", "RUI02", "RUI03", "RUI04", "RUI05", "RUI06", "RUI07", "RUI08", "RUI09"),
        FED751 = cms.untracked.vstring("RUI10", "RUI11", "RUI12", "RUI13", "RUI14", "RUI15", "RUI16", "RUI17", "RUI18"),
        FED752 = cms.untracked.vstring("RUI19", "RUI20", "RUI21", "RUI22", "RUI23", "RUI24", "RUI25", "RUI26", "RUI27"),
        FED753 = cms.untracked.vstring("RUI28", "RUI29", "RUI30", "RUI31", "RUI32", "RUI33", "RUI34", "RUI35", "RUI36")
        # For CSC TF.
        #FED760 = cms.untracked.vstring("RUI00")
    )
)
