import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripApvSimulationParametersRcd'),
        tag = cms.string('SiStripApvSimulationParameters_2016preVFP_v1')
    ))
)

process.apvSimParam = cms.ESSource("SiStripApvSimulationParametersESSource",
    apvBaselines_nBinsPerBaseline=cms.untracked.uint32(82),
    apvBaselines_minBaseline=cms.untracked.double(0.),
    apvBaselines_maxBaseline=cms.untracked.double(738.),
    apvBaselines_puBinEdges=cms.untracked.vdouble(0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18., 20., 22., 24., 26., 28., 30., 32., 34., 36., 38., 40., 42., 44., 46., 48., 50.),
    apvBaselines_zBinEdges=cms.untracked.vdouble(0., 10., 20., 30., 40., 50., 60., 70., 90.),
    apvBaselines_rBinEdges_TID=cms.untracked.vdouble(0., 10., 20., 30., 40., 50., 60., 70., 90.),
    apvBaselines_rBinEdges_TEC=cms.untracked.vdouble(0., 10., 20., 30., 40., 50., 60., 70., 90.),
    apvBaselinesFile_tib1=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TIB1_11us.txt"),
    apvBaselinesFile_tib2=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TIB2_14us.txt"),
    apvBaselinesFile_tib3=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TIB3_15us.txt"),
    apvBaselinesFile_tib4=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TIB4_18us.txt"),
    apvBaselinesFile_tob1=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TOB1_10us.txt"),
    apvBaselinesFile_tob2=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TOB2_12us.txt"),
    apvBaselinesFile_tob3=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TOB3_15us.txt"),
    apvBaselinesFile_tob4=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TOB4_19us.txt"),
    apvBaselinesFile_tob5=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TOB5_24us.txt"),
    apvBaselinesFile_tob6=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TOB6_25us.txt"),
    apvBaselinesFile_tid1=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TID1_9us.txt"),
    apvBaselinesFile_tid2=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TID2_9us.txt"),
    apvBaselinesFile_tid3=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TID3_9us.txt"),
    apvBaselinesFile_tec1=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TEC1_10us.txt"),
    apvBaselinesFile_tec2=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TEC2_11us.txt"),
    apvBaselinesFile_tec3=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TEC3_11us.txt"),
    apvBaselinesFile_tec4=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TEC4_13us.txt"),
    apvBaselinesFile_tec5=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TEC5_14us.txt"),
    apvBaselinesFile_tec6=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TEC6_15us.txt"),
    apvBaselinesFile_tec7=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TEC7_16us.txt"),
    apvBaselinesFile_tec8=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TEC8_16us.txt"),
    apvBaselinesFile_tec9=cms.untracked.FileInPath("SimTracker/SiStripDigitizer/data/APVBaselines_TEC9_16us.txt")
    )
process.prod = cms.EDAnalyzer("SiStripApvSimulationParametersBuilder")

process.p = cms.Path(process.prod)
