import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:PhysicsPerformance.db'


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(10)
                            )

# process.PoolDBOutputService.DBParameters.messageLevel = 3


process.mywriter = cms.EDAnalyzer("ProducePFCalibrationObject",
                                  write = cms.untracked.bool(False),
                                  toWrite = cms.VPSet(cms.PSet(fType      = cms.untracked.string("PFfa_BARREL"), # PFfa_BARREL - PFfa_ENDCAP ....
                                                               formula    = cms.untracked.string("[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])"),
                                                               limits     = cms.untracked.vdouble(0., 1000.),
                                                               parameters = cms.untracked.vdouble(1.10772, 0.186273, -0.47812, 62.5754, 1.31965, 35.2559)
                                                               )
                                                      ),
                                  read = cms.untracked.bool(True),
                                  toRead = cms.untracked.vstring("PFfa_BARREL") # same strings as fType
                                  )


process.p = cms.Path(process.mywriter)

from CondCore.DBCommon.CondDBCommon_cfi import CondDBCommon
CondDBCommon.connect = "sqlite_file:PFCalibration.db"

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                  CondDBCommon,
                                  toPut = cms.VPSet(cms.PSet(record = cms.string('PFCalibrationRcd'),
                                                             tag = cms.string('PFCalibration'),
                                                             timetype   = cms.untracked.string('runnumber')
                                                             )
                                                             ),
                                  loadBlobStreamer = cms.untracked.bool(False),
                                  #    timetype   = cms.untracked.string('lumiid')
                                  #    timetype   = cms.untracked.string('runnumber')
                                  )

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START311_V1A::All'
#process.GlobalTag.connect   = 'sqlite_file:/afs/cern.ch/user/c/cerminar/public/Alca/GlobalTag/GR_R_311_V2.db'

process.GlobalTag.toGet = cms.VPSet(
  cms.PSet(record = cms.string("PFCalibrationRcd"),
           tag = cms.string("PFCalibration"),
           connect = cms.untracked.string("sqlite_file:PFCalibration.db")
           #connect = cms.untracked.string("sqlite_file:PFCalibration.db")
          )
)
