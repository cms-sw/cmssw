import FWCore.ParameterSet.Config as cms

process = cms.Process("DATACONVERTER")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.load('CondCore.CondDB.CondDB_cfi')

process.load("Configuration.Geometry.GeometryDB_cff")

#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

#process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

#process.load("CondCore.DBCommon.CondDBSetup_cfi")

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
import CalibTracker.Configuration.Common.PoolDBESSource_cfi

process.conditionsInTrackerAlignmentRcd = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
     connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/data/commonValidation/vbotta/misalignments/check/CMSSW_11_0_0_pre13/src/Alignment/TrackerAlignment/test/geometry_MisalignmentScenario_BPIXModulesHB1_PhiYGlob_Min30umrad_from105X_upgrade2018_design_v3.db'),
     toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                               tag = cms.string('Alignments')
        # ),cms.PSet(record = cms.string('TrackerTopologyRcd'),
        #                       tag = cms.string('Alignments')
                               )
                      )
    )
process.prefer_conditionsInTrackerAlignmentRcd = cms.ESPrefer("PoolDBESSource", "conditionsInTrackerAlignmentRcd")


#process.PoolDBOutputService = cms.Service("PoolDBOutputService",
#    process.CondDBSetup,
#    toPut = cms.VPSet(cms.PSet(
#        record = cms.string('TrackerAlignmentRcd'),
#        tag = cms.string('TibTidTecAllSurvey_v2')
#    ), 
#        cms.PSet(
#            record = cms.string('TrackerAlignmentErrorExtendedRcd'),
#            tag = cms.string('TibTidTecAllSurveyAPE_v2')
##        ),
##        cms.PSet(
##        record = cms.string('TrackerTopologyRcd'),
##        tag = cms.string('TibTidTecAllSurveyAPE_v2')
#        )),
#
#   connect = cms.string('sqlite_file:TibTidTecAllSurvey.db')                                     
#)


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
#process.GlobalTag.globaltag = 'IDEAL'

process.GlobalTag.globaltag = '120X_mcRun3_2021_design_Queue'
#process.GlobalTag.globaltag = '105X_upgrade2018_design_v3'
