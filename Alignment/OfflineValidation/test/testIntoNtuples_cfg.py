import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = "GR_R_53_V16D::All" 
#process.GlobalTag.globaltag = "START61_V11::All" 
process.GlobalTag.globaltag = "GR_R_61_V7::All" 
#process.GlobalTag.globaltag = "FT_R_53_V6C::All" 

#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Configuration.Geometry.GeometryDB_cff")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")


process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    ),
    files = cms.untracked.PSet(
        detailedInfo = cms.untracked.PSet(

        )
    )
)

from CondCore.DBCommon.CondDBSetup_cfi import *

process.ZeroGeom = cms.ESSource("PoolDBESSource",CondDBSetup,
		connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1260/jobData/jobm2/alignments_MP.db'), 
		timetype = cms.string("runnumber"),
		toGet = cms.VPSet(
			cms.PSet(record = cms.string('TrackerAlignmentRcd'), 
				tag = cms.string('Alignments')
				),
			), 
		)
							
process.ZeroAPE = cms.ESSource("PoolDBESSource",CondDBSetup,
		connect = cms.string('frontier://FrontierProd/CMS_COND_31X_FROM21X'),
		timetype = cms.string("runnumber"),
		toGet = cms.VPSet(
			cms.PSet(
				record = cms.string('TrackerAlignmentErrorExtendedRcd'),
				tag = cms.string('TrackerIdealGeometryErrors210_mc')
				)
			)		
		)

process.trackerBowedSensors = cms.ESSource("PoolDBESSource",CondDBSetup,
		connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1260/jobData/jobm2/alignments_MP.db'),
		toGet = cms.VPSet(
			cms.PSet(record = cms.string('TrackerSurfaceDeformationRcd'),
				tag = cms.string('Deformations')
				)
			)
		)

#process.prefer("GlobalTag")
process.es_prefer_ZeroGeom = cms.ESPrefer("PoolDBESSource", "ZeroGeom")
process.es_prefer_ZeroAPE = cms.ESPrefer("PoolDBESSource", "ZeroAPE")
process.es_prefer_trackerBowedSensors = cms.ESPrefer("PoolDBESSource", "trackerBowedSensors")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
		input = cms.untracked.int32(1)
		)

process.dump = cms.EDAnalyzer("TrackerGeometryIntoNtuples",
		#outputFile = cms.untracked.string('myInputGeometry_mp1260m2.root'),
		outputFile = cms.untracked.string('myInputGeometry_mp1260m2_14feb2013.root'),
		#outputFile = cms.untracked.string('myInputGeometry_2012AB.root'),
		outputTreename = cms.untracked.string('alignTree')
		)

process.p = cms.Path(process.dump)


