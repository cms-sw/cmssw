import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
###################################################################
# Setup 'standard' options
###################################################################
options = VarParsing.VarParsing()
options.register('Scenario',
                 _settings.DEFAULT_VERSION, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "geometry version to use")
options.parseArguments()

###################################################################
# get Global Tag and ERA
###################################################################
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(options.Scenario)
process = cms.Process("Alignment", ERA)

process.load("Configuration.StandardSequences.MagneticField_cff") # B-field map
if(options.Scenario == _settings.DEFAULT_VERSION):
    print("Loading default scenario: ", _settings.DEFAULT_VERSION)
    process.load('Configuration.Geometry.GeometryExtendedRun4DefaultReco_cff')
else:
    process.load('Configuration.Geometry.GeometryExtended'+options.Scenario+'Reco_cff')    
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff") # Global tag


################################################################################
# parameters to configure:
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,GLOBAL_TAG)

process.load("Alignment.TrackerAlignment.createIdealTkAlRecords_cfi")
process.createIdealTkAlRecords.alignToGlobalTag = False
################################################################################

usedGlobalTag = process.GlobalTag.globaltag.value()
print("Using Global Tag:", usedGlobalTag)

from CondCore.CondDB.CondDB_cfi import *
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    CondDB,
    timetype = cms.untracked.string("runnumber"),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string("TrackerAlignmentRcd"),
            tag = cms.string("Alignments")
        ),
        cms.PSet(
            record = cms.string("TrackerAlignmentErrorExtendedRcd"),
            tag = cms.string("AlignmentErrorsExtended")
        ),
        cms.PSet(
            record = cms.string("TrackerSurfaceDeformationRcd"),
            tag = cms.string("AlignmentSurfaceDeformations")
        ),
    )
)
process.PoolDBOutputService.connect = \
    ("sqlite_file:tracker_alignment_payloads_"+
     options.Scenario+("_reference.db"
                    if process.createIdealTkAlRecords.createReferenceRcd
                    else "_fromIdealGeometry.db"))

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.source = cms.Source("EmptySource")

process.p = cms.Path(process.createIdealTkAlRecords)
