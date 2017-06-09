import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
#process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("CondCore.CondDB.CondDB_cfi")
#process.CondDBCommon.connect = 'sqlite_file:PhysicsPerformance.db'


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
                     formula    = cms.untracked.string("[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(-13.9219, 14.9124, 5.38578, 0.861981, -0.00759275, 0.00373563, -1.17946, -1.69561, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfb_BARREL"),
                     formula    = cms.untracked.string("[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(2.18802, 0.522053, -4.67354, 12.109, 1.75318, 0.187919, -6.26234, -0.607392, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfc_BARREL"),
                     formula    = cms.untracked.string("[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(1.46854, 0.830152, -5.86601, 2.08229, 0.575016, 0.0291232, 0.364802, -1.50142, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfaEta_BARRELEH"),
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(0.0185555, -0.0470674, 396.959, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfbEta_BARRELEH"),
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(0.0396458, 0.114128, 251.405, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfaEta_BARRELH"),
                     formula    = cms.untracked.string("[0]+[1]*x"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(0.00434994, -5.16564e-06, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfbEta_BARRELH"),
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(-0.0232604, 0.0937525, 34.9935, )
                    ),

            cms.PSet(fType      = cms.untracked.string("PFfa_ENDCAP"),
                     formula    = cms.untracked.string("[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(0.962468, 11.9536, -27.7088, 0.755474, 0.0791012, 0.0011082, 0.158734, -2.1, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfb_ENDCAP"),
                     formula    = cms.untracked.string("[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(-0.43671, 2.90096, -5.10099, 1.20771, -1.30656, 0.0189607, 0.270027, -2.30372, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfc_ENDCAP"),
                     formula    = cms.untracked.string("[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(1.11623, 1.66795, -5.49393, 7.73734, 0.259307, 0.0708707, 0.950338, -0.922411, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfaEta_ENDCAPEH"),
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(384.307, -384.305, 1.14535e+07, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfbEta_ENDCAPEH"),
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(0.0282454, -0.107485, 96.1833, )
                    ),

            cms.PSet(fType      = cms.untracked.string("PFfaEta_ENDCAPH"),
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])+[3]*[3]*exp(-x*x/([4]*[4]))"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(-0.0106029, -0.692207, 0.0542991, -0.171435, -61.2277, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfbEta_ENDCAPH"),
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])+[3]*[3]*exp(-x*x/([4]*[4]))"),
                     limits     = cms.untracked.vdouble(1., 1000.),
                     parameters = cms.untracked.vdouble(0.0514927, -0.291726, 5.32989, 0.309057, -103.359, )
                    ),
            ),


  read = cms.untracked.bool(True),
  toRead = cms.untracked.vstring("PFfa_BARREL",
                                 "PFfb_BARREL",
                                 "PFfc_BARREL",
                                 "PFfa_ENDCAP",
                                 "PFfb_ENDCAP",
                                 "PFfc_ENDCAP",
                                 #### New Functions
                                 "PFfaEta_BARRELEH",
                                 "PFfbEta_BARRELEH",
                                 "PFfaEta_ENDCAPEH",
                                 "PFfbEta_ENDCAPEH",
                                 "PFfaEta_BARRELH",
                                 "PFfbEta_BARRELH",
                                 "PFfaEta_ENDCAPH",
                                 "PFfbEta_ENDCAPH",
                                 #### Left older functions untouched for backward compatibility
                                 "PFfaEta_BARREL",
                                 "PFfbEta_BARREL",
                                 "PFfaEta_ENDCAP",
                                 "PFfbEta_ENDCAP",

                                 ) # same strings as fType
)


process.p = cms.Path(process.mywriter)

from CondCore.DBCommon.CondDBCommon_cfi import CondDBCommon
CondDBCommon.connect = "sqlite_file:PFCalibration.db"

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                  CondDBCommon,
                                  toPut = cms.VPSet(cms.PSet(record = cms.string('PFCalibrationRcd'),
                                                             tag = cms.string('PFCalibration_v6_mc'),
                                                             timetype   = cms.untracked.string('runnumber')
                                                             )
                                                             ),
                                  loadBlobStreamer = cms.untracked.bool(False),
                                  #    timetype   = cms.untracked.string('lumiid')
                                  #    timetype   = cms.untracked.string('runnumber')
                                  )

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = '90X_upgrade2017_realistic_v20'
#process.GlobalTag.connect   = 'sqlite_file:/afs/cern.ch/user/c/cerminar/public/Alca/GlobalTag/GR_R_311_V2.db'

process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("PFCalibrationRcd"),
             tag = cms.string("PFCalibration_v6_mc"),
             connect = cms.string("sqlite_file:PFCalibration.db")
             #connect = cms.untracked.string("sqlite_file:PFCalibration.db")
             )
    )
