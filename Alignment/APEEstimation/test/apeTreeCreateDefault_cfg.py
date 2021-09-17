########################################################################
###
###             Read out APEs from .db files and convert them to trees 
###             that can be read by the APE validation plot tools.
###
###             Intended to provide a straightforward comparison of 
###             measured APE values to values stored in .db files
###
########################################################################
###
###             HOW TO USE:
###                     1. Run the default setup procedure for the APE 
###                        tool (including creation of a TrackerTree)
###                     2. Configure the apeTreeCreateDefault tool below
###                        and run it with cmsRun
###                     3. Use output file in validation, for example in 
###                        macros/commandsDrawComparison.C
###
########################################################################

import FWCore.ParameterSet.Config as cms
from Alignment.APEEstimation.SectorBuilder_cff import *
import os
##
## User options
##

# Run number to use for data in case one uses a multi-IOV object
theFirstRun = 1

# Which global tag to use
theGlobalTag = 'auto:phase1_2017_realistic'

# Source from which to get the APE object
theSource = 'frontier://FrontierProd/CMS_CONDITIONS'

# Tag to extract the APE object
theTag = 'TrackerAlignmentExtendedErrors_Upgrade2017_pseudoAsymptotic_v3'

# Name and path of output File
theOutputFile = 'defaultAPE.root'

# Sector definitions, RecentSectors is the typical granularity
theSectors = RecentSectors

##
## Process definition
##
process = cms.Process("ApeTreeCreateDefault")



##
## Message Logger
##
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.DefaultAPETree=dict()
process.MessageLogger.SectorBuilder=dict()
process.MessageLogger.cerr.INFO.limit = 0
process.MessageLogger.cerr.default.limit = 0
process.MessageLogger.cerr.DefaultAPETree = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.SectorBuilder = cms.untracked.PSet(limit = cms.untracked.int32(-1))



##
## Process options
##
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
)



##
## Input Files
##
process.source = cms.Source("EmptySource", firstRun = cms.untracked.uint32(theFirstRun))



##
## Number of Events
##
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

### Load desired default APEs
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, theGlobalTag, '')

from CondCore.CondDB.CondDB_cfi import *
CondDBAlignmentError = CondDB.clone(connect = cms.string(theSource))
process.myTrackerAlignmentErr = cms.ESSource("PoolDBESSource",
    CondDBAlignmentError,
    timetype = cms.string("runnumber"),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('TrackerAlignmentErrorExtendedRcd'),
            tag = cms.string(theTag)
        )
    )
)
process.es_prefer_trackerAlignmentErr = cms.ESPrefer("PoolDBESSource","myTrackerAlignmentErr")



##
## Define Sequence
##
process.ApeTreeCreateDefaultSequence = cms.Sequence()

process.ApeTreeCreateDefault = cms.EDAnalyzer('ApeTreeCreateDefault',
    resultFile = cms.string(theOutputFile),
    trackerTreeFile = cms.string(os.environ['CMSSW_BASE'] + '/src/Alignment/TrackerAlignment/hists/TrackerTree.root'),
    sectors = theSectors,
)
                                                              

process.ApeTreeCreateDefaultSequence *= process.ApeTreeCreateDefault



##
## Path
##
process.p = cms.Path(
    process.ApeTreeCreateDefaultSequence
)






