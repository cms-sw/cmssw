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


process.mywriter = cms.EDAnalyzer(
  "ProducePFCalibrationObject",
  write = cms.untracked.bool(False),
  toWrite = cms.VPSet(
            cms.PSet(fType      = cms.untracked.string("PFfa_BARREL"), 
                     formula    = cms.untracked.string("[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(1.15665, 0.165627, 0.827718, 231.339, 2.45332, 29.6603,  ) 
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfb_BARREL"), 
                     formula    = cms.untracked.string("[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(0.994603, 0.13632, -0.758013, 183.627, 1, 39.6784,  ) 
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfc_BARREL"), 
                     formula    = cms.untracked.string("[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(0.956544, 0.0857207, -0.44347, 63.3479, 1.24174, 12.322,  ) 
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfaEta_BARREL"), 
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(0.014664, -0.0426776, 431.054,  ) 
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfbEta_BARREL"), 
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(0.00975451, 0.102247, 436.21,  ) 
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfa_ENDCAP"), 
                     formula    = cms.untracked.string("[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(1.1272, 0.258536, 0.808071, 214.039, 2, 47.2602,  ) 
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfb_ENDCAP"), 
                     formula    = cms.untracked.string("[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(0.982824, 0.0977533, 0.155416, 240.379, 1.2, 78.3083,  ) 
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfc_ENDCAP"), 
                     formula    = cms.untracked.string("[0]+([1]+[2]/sqrt(x))*exp(-x/[3])-[4]*exp(-x*x/[5])"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(0.950244, 0.00564779, 0.227162, 207.786, 1.32824, 22.1825,  ) 
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfaEta_ENDCAP"), 
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(-0.000582903, -0.000482148, 209.466,  ) 
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfbEta_ENDCAP"), 
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])+[3]*[3]*exp(-x*x/([4]*[4]))"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(0.0267319, -0.554552, 1.71188, 0.235834, -135.431,  ) 
                    ),
            ),
  read = cms.untracked.bool(True),
  toRead = cms.untracked.vstring("PFfa_BARREL",
                                 "PFfa_ENDCAP",
                                 "PFfb_BARREL",
                                 "PFfb_ENDCAP",
                                 "PFfc_BARREL",
                                 "PFfc_ENDCAP",
                                 "PFfaEta_BARREL",
                                 "PFfaEta_ENDCAP",
                                 "PFfbEta_BARREL",
                                 "PFfbEta_ENDCAP") # same strings as fType
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
