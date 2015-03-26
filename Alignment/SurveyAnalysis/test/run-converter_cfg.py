import FWCore.ParameterSet.Config as cms

process = cms.Process("DATACONVERTER")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("Alignment.SurveyAnalysis.SurveyInfoScenario_cff")

process.MessageLogger = cms.Service("MessageLogger",
    test = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        noLineBreaks = cms.untracked.bool(True)
    ),
    statistics = cms.untracked.vstring('cout', 
        'test'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        noLineBreaks = cms.untracked.bool(True)
    ),
    categories = cms.untracked.vstring('Alignment'),
    destinations = cms.untracked.vstring('cout', 
        'test')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBSetup,
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('TibTidTecAllSurvey_v2')
    ), 
        cms.PSet(
            record = cms.string('TrackerAlignmentErrorExtendedRcd'),
            tag = cms.string('TibTidTecAllSurveyAPE_v2')
        )),
   connect = cms.string('sqlite_file:TibTidTecAllSurvey.db')                                     
)

process.mydataconverter = cms.EDAnalyzer("SurveyDataConverter",
    applyFineInfo = cms.bool(True),
    MisalignmentScenario = cms.PSet(
        process.SurveyInfoScenario
    ),
    applyErrors = cms.bool(True),
    textFileNames = cms.PSet(
        forTID = cms.untracked.string('../data/GetTibSurvey/TIDNewSurvey.dat'),
        forTIB = cms.untracked.string('../data/GetTibSurvey/TIBNewSurvey.dat')
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
process.GlobalTag.globaltag = 'IDEAL_V9::All'
