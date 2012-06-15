######################################################################
######################################################################
intoNTuplesTemplate="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("ValidationIntoNTuples")

# global tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo." 

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

#process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_cff")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo', 
        'cout')
) 

#removed: APE
#removed: dbLoad
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.GeomToComp = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
connect = cms.string('.oO[dbpath]Oo.'),

    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('.oO[tag]Oo.')
    ))
   
)
process.es_prefer_geom=cms.ESPrefer("PoolDBESSource","GeomToComp")

from CondCore.DBCommon.CondDBSetup_cfi import *

process.ZeroAPE = cms.ESSource("PoolDBESSource",CondDBSetup,
								connect = cms.string('frontier://FrontierProd/CMS_COND_31X_FROM21X'),
								timetype = cms.string("runnumber"),
								toGet = cms.VPSet(
											cms.PSet(
												record = cms.string('TrackerAlignmentErrorRcd'),
												tag = cms.string('TrackerIdealGeometryErrors210_mc')
											))
								)
process.es_prefer_ZeroAPE = cms.ESPrefer("PoolDBESSource", "ZeroAPE")


process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.dump = cms.EDAnalyzer("TrackerGeometryIntoNtuples",
    outputFile = cms.untracked.string('.oO[workdir]Oo./.oO[alignmentName]Oo.ROOTGeometry.root'),
    outputTreename = cms.untracked.string('alignTree')
)

process.p = cms.Path(process.dump)  
"""


######################################################################
######################################################################
compareTemplate="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("validation")

# global tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo." 
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

#process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_cff")
# the input .GlobalPosition_Frontier_cff is providing the frontier://FrontierProd/CMS_COND_31X_ALIGNMENT in the release which does not provide the ideal geometry
#process.GlobalPosition.connect = 'frontier://FrontierProd/CMS_COND_31X_FROM21X'

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo', 
        'cout')
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

  # configuration of the Tracker Geometry Comparison Tool
  # Tracker Geometry Comparison
process.load("Alignment.OfflineValidation.TrackerGeometryCompare_cfi")
  # the input "IDEAL" is special indicating to use the ideal geometry of the release

process.TrackerGeometryCompare.inputROOTFile1 = '.oO[comparedGeometry]Oo.'
process.TrackerGeometryCompare.inputROOTFile2 = '.oO[referenceGeometry]Oo.'
process.TrackerGeometryCompare.outputFile = ".oO[workdir]Oo./.oO[name]Oo..Comparison_common.oO[common]Oo..root"
process.TrackerGeometryCompare.levels = [ .oO[levels]Oo. ]

  ##FIXME!!!!!!!!!
  ##replace TrackerGeometryCompare.writeToDB = .oO[dbOutput]Oo.
  ##removed: dbOutputService

process.p = cms.Path(process.TrackerGeometryCompare)
"""
  

######################################################################
######################################################################
dbOutputTemplate= """
//_________________________ db Output ____________________________
        # setup for writing out to DB
        include "CondCore/DBCommon/data/CondDBSetup.cfi"
#       include "CondCore/DBCommon/data/CondDBCommon.cfi"

    service = PoolDBOutputService {
        using CondDBSetup
        VPSet toPut = {
            { string record = "TrackerAlignmentRcd"  string tag = ".oO[tag]Oo." },
            { string record = "TrackerAlignmentErrorRcd"  string tag = ".oO[errortag]Oo." }
        }
                string connect = "sqlite_file:.oO[workdir]Oo./.oO[name]Oo.Common.oO[common]Oo..db"
                # untracked string catalog = "file:alignments.xml"
        untracked string timetype = "runnumber"
    }
"""

