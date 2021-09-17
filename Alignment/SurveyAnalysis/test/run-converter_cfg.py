import FWCore.ParameterSet.Config as cms

process = cms.Process("DATACONVERTER")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'auto:phase1_2021_realistic'

process.load("Configuration.Geometry.GeometryDB_cff")
process.load("Alignment.SurveyAnalysis.SurveyInfoScenario_cff")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        enableStatistics = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    files = cms.untracked.PSet(
        test = cms.untracked.PSet(
            enableStatistics = cms.untracked.bool(True),
            noLineBreaks = cms.untracked.bool(True),
            threshold = cms.untracked.string('DEBUG')
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = cms.string('sqlite_file:TibTidTecAllSurvey.db')
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                                                                     tag = cms.string('TibTidTecAllSurvey_v2')
                                                                     ), 
                                                            cms.PSet(record = cms.string('TrackerAlignmentErrorExtendedRcd'),
                                                                     tag = cms.string('TibTidTecAllSurveyAPE_v2')
                                                                     ),
                                                           )
                                          )


process.mydataconverter = cms.EDAnalyzer("SurveyDataConverter",
    applyFineInfo = cms.bool(True),
    MisalignmentScenario = cms.PSet(
        process.SurveyInfoScenario
    ),
    applyErrors = cms.bool(True),
    textFileNames = cms.PSet(
        forTID = cms.untracked.string('./TIDSurvey.dat'),
        forTIB = cms.untracked.string('./TIBSurvey.dat')
    ),
    applyCoarseInfo = cms.bool(True),                                  
    TOBerrors = cms.vdouble(0.014, 0.05, 0.02, 0.003),
    TECerrors = cms.vdouble(0.06, 0.015, 0.007, 0.002),
    TIDerrors = cms.vdouble(0.045, 0.035, 0.0185, 0.0054),
    TIBerrors = cms.vdouble(0.075, 0.045, 0.018)
)

# process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.mydataconverter)
# process.ep = cms.EndPath(process.print)

