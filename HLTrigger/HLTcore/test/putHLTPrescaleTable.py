import FWCore.ParameterSet.Config as cms

process = cms.Process("PUT")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.HLTPrescaleRecorder=dict()

process.source = cms.Source("EmptySource",
   firstRun = cms.untracked.uint32(1)
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.load("HLTrigger.HLTcore.hltPrescaleRecorder_cfi")
process.hltPrescaleRecorder.src=-1
process.hltPrescaleRecorder.condDB=cms.bool(True)

#import CondCore.DBCommon.CondDBSetup_cfi
#process.PoolDBESSource = cms.ESSource("PoolDBESSource",
#   CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
#   connect = cms.string('sqlite_file:HLTPrescaleTable.db'),
#   toGet = cms.VPSet(
#    cms.PSet(
#     record=cms.string("HLTPrescaleTableRcd"),
#     tag = cms.string("/dev/null")
#    )
#   )
#)

import CondCore.DBCommon.CondDBSetup_cfi
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
   CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
   connect = cms.string('sqlite_file:HLTPrescaleTable.db'),
   timetype = cms.untracked.string("timestamp"),
   toPut = cms.VPSet(
    cms.PSet(
     record=cms.string("HLTPrescaleTableRcd"),
     tag = cms.string("/dev/null")
    )
   )
)

process.PrescaleTable = cms.PSet()
from HLTrigger.HLTcore.tmpPrescaleService import PrescaleService
for pn, pv in PrescaleService.parameters_().items(): setattr(process.PrescaleTable, pn, pv)
process.hltPrescaleRecorder.psetName=cms.string("PrescaleTable")

from HLTrigger.HLTcore.tmpPrescaleService import HLTConfigVersion
process.HLTConfigVersion = HLTConfigVersion
#process.PoolDBESSource.toGet[0].tag = HLTConfigVersion.tableName
process.PoolDBOutputService.toPut[0].tag = HLTConfigVersion.tableName

process.p = cms.Path(process.hltPrescaleRecorder)
