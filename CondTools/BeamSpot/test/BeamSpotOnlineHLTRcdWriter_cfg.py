import FWCore.ParameterSet.Config as cms

process = cms.Process("write2DB")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
from CondCore.CondDB.CondDB_cfi import *

#################################
# Produce a SQLITE FILE
#
CondDBBeamSpotObjects = CondDB.clone(connect = cms.string('sqlite_file:test.db')) # choose an output name

#################################
#
# upload conditions to orcon
#
#process.CondDBCommon.connect = "oracle://cms_orcoff_prep/CMS_COND_BEAMSPOT"
#process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = 'MC_31X_V2::All'

#################################

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          #process.CondDBCommon,
                                          CondDBBeamSpotObjects,
                                          timetype = cms.untracked.string('lumiid'), #('lumiid'), #('runnumber')
                                          toPut = cms.VPSet(cms.PSet(
    record = cms.string('BeamSpotOnlineHLTObjectsRcd'), # BeamSpotOnlineHLT record
    tag = cms.string('BSHLT_tag') )),                    # choose your favourite tag
    loadBlobStreamer = cms.untracked.bool(False)
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(1)
                    )

process.beamspotonlinewriter = cms.EDAnalyzer("BeamSpotOnlineHLTRcdWriter",
                        InputFileName = cms.untracked.string('../data/BeamFitResults_Run306171.txt'),  # choose your input file
                        #IOVStartRun = cms.untracked.uint32(306171), # Customize your Run
                        #IOVStartLumi = cms.untracked.uint32(497),   # Customize your Lumi
                   )

process.p = cms.Path(process.beamspotonlinewriter)