import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
# Message logger service
process.load("FWCore.MessageService.MessageLogger_cfi")

########## Original ###########
## Ideal geometry
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
#
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
###############################

# MTD Geometry and phase 2 configuration
process.load("Configuration.Geometry.GeometryExtendedRun4DefaultReco_cff")

import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep


# Reading from DB
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('MTDAlignmentRcd'),
        tag = cms.string('Alignments')
    ), 
        cms.PSet(
            record = cms.string('MTDAlignmentErrorExtendedRcd'),
            tag = cms.string('AlignmentErrorsExtended')
        )),
    #connect = cms.string('sqlite_file:AlignmentsNominal.db')
    connect = cms.string('sqlite_file:AlignmentsScenario.db')
)



process.prod = cms.EDAnalyzer("TestMTDReader")

process.p1 = cms.Path(process.prod)
process.MessageLogger.cerr.default.limit = 100000
process.MessageLogger.cerr.INFO.limit = -1


