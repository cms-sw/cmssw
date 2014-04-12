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

.oO[condLoad]Oo.

process.source = cms.Source("EmptySource",
    firstRun=cms.untracked.uint32(.oO[runGeomComp]Oo.)
    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.dump = cms.EDAnalyzer("TrackerGeometryIntoNtuples",
    # outputFile = cms.untracked.string('.oO[workdir]Oo./.oO[alignmentName]Oo.ROOTGeometry.root'),
    outputFile = cms.untracked.string('.oO[alignmentName]Oo.ROOTGeometry.root'),
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

process.load("DQM.SiStripCommon.TkHistoMap_cfi")

process.DQMStore=cms.Service("DQMStore")
#process.TkDetMap = cms.Service("TkDetMap")
#process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

process.load("DQMServices.Core.DQMStore_cfg") 
#process.DQMStore=cms.Service("DQMStore")

  # configuration of the Tracker Geometry Comparison Tool
  # Tracker Geometry Comparison
process.load("Alignment.OfflineValidation.TrackerGeometryCompare_cfi")
  # the input "IDEAL" is special indicating to use the ideal geometry of the release

process.TrackerGeometryCompare.inputROOTFile1 = '.oO[comparedGeometry]Oo.'
process.TrackerGeometryCompare.inputROOTFile2 = '.oO[referenceGeometry]Oo.'
# process.TrackerGeometryCompare.outputFile = ".oO[workdir]Oo./.oO[name]Oo..Comparison_common.oO[common]Oo..root"
process.TrackerGeometryCompare.outputFile = ".oO[name]Oo..Comparison_common.oO[common]Oo..root"

process.load("CommonTools.UtilAlgos.TFileService_cfi")  
#process.TFileService = cms.Service("TFileService",
#		fileName = cms.string('TkSurfDeform.root') 
#		)
process.TFileService.fileName = cms.string("TkSurfDeform_.oO[name]Oo..Comparison_common.oO[common]Oo..root") 

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
                # string connect = "sqlite_file:.oO[workdir]Oo./.oO[name]Oo.Common.oO[common]Oo..db"
                string connect = "sqlite_file:.oO[name]Oo.Common.oO[common]Oo..db"
                # untracked string catalog = "file:alignments.xml"
        untracked string timetype = "runnumber"
    }
"""

