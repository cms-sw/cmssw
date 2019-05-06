import FWCore.ParameterSet.Config as cms



################################################################
#### This config is used to generate payload for PFCalibration 
################################################################


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
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(-30.7141, 31.7583, 4.40594, 1.70914, 0.0613696, 0.000104857, -1.38927, -0.743082, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfb_BARREL"),
                     formula    = cms.untracked.string("[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(2.25366, 0.537715, -4.81375, 12.109, 1.80577, 0.187919, -6.26234, -0.607392, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfc_BARREL"),
                     formula    = cms.untracked.string("[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(1.5126, 0.855057, -6.04199, 2.08229, 0.592266, 0.0291232, 0.364802, -1.50142, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfaEta_BARRELEH"),
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(0.0185555, -0.0470674, 396.959, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfbEta_BARRELEH"),
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(0.0396458, 0.114128, 251.405, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfaEta_BARRELH"),
                     formula    = cms.untracked.string("[0]+[1]*x"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(0.00434994, -5.16564e-06, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfbEta_BARRELH"),
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(-0.0232604, 0.0937525, 34.9935, )
                    ),

            cms.PSet(fType      = cms.untracked.string("PFfa_ENDCAP"),
                     formula    = cms.untracked.string("[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(1.17227, 13.1489, -29.1672, 0.604223, 0.0426363, 3.30898e-15, 0.165293, -7.56786, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfb_ENDCAP"),
                     formula    = cms.untracked.string("[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(-0.974251, 1.61733, 0.0629183, 7.78495, -0.774289, 7.81399e-05, 0.139116, -4.25551, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfc_ENDCAP"),
                     formula    = cms.untracked.string("[0]+((([1]+([2]/sqrt(x)))*exp(-(x^[6]/[3])))-([4]*exp(-(x^[7]/[5]))))"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(1.01863, 1.29787, -3.97293, 21.7805, 0.810195, 0.234134, 1.42226, -0.0997326, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfaEta_ENDCAPEH"),
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(0.0112692, -2.68063, 2.90973, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfbEta_ENDCAPEH"),
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(-0.0192991, -0.265, 80.5502, )
                    ),

            cms.PSet(fType      = cms.untracked.string("PFfaEta_ENDCAPH"),
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])+[3]*[3]*exp(-x*x/([4]*[4]))"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(-0.0106029, -0.692207, 0.0542991, -0.171435, -61.2277, )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfbEta_ENDCAPH"),
                     formula    = cms.untracked.string("[0]+[1]*exp(-x/[2])+[3]*[3]*exp(-x*x/([4]*[4]))"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(0.0214894, -0.266704, 5.2112, 0.303578, -104.367, )
                    ),
#########################################################################################################
            cms.PSet(fType      = cms.untracked.string("PFfcEta_BARRELH"),
                     formula    = cms.untracked.string("[3]*((x-[0])^[1])+[2]"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(0., 2., 0., 1., )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfcEta_ENDCAPH"),
                     formula    = cms.untracked.string("[3]*((x-[0])^[1])+[2]"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(0., 0., 0.05, 0., )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfdEta_ENDCAPH"),
                     formula    = cms.untracked.string("[3]*((x-[0])^[1])+[2]"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(1.5, 4., -1.1, 1., )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfcEta_BARRELEH"),
                     formula    = cms.untracked.string("[3]*((x-[0])^[1])+[2]"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(0., 2., 0., 1., )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfcEta_ENDCAPEH"),
                     formula    = cms.untracked.string("[3]*((x-[0])^[1])+[2]"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(0., 0., 0., 0., )
                    ),
            cms.PSet(fType      = cms.untracked.string("PFfdEta_ENDCAPEH"),
                     formula    = cms.untracked.string("[3]*((x-[0])^[1])+[2]"),
                     limits     = cms.untracked.vdouble(0., 1000.),
                     parameters = cms.untracked.vdouble(1.5, 2., 0.6, 1., )
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
                                 
                                 #### New functions by Bhumika on September 2018

                                 "PFfcEta_BARRELH",
                                 "PFfcEta_ENDCAPH",
                                 "PFfdEta_ENDCAPH",
                                 "PFfcEta_BARRELEH",
                                 "PFfcEta_ENDCAPEH",
                                 "PFfdEta_ENDCAPEH",


                                 ) # same strings as fType
)


process.p = cms.Path(process.mywriter)

from CondCore.DBCommon.CondDBCommon_cfi import CondDBCommon
CondDBCommon.connect = "sqlite_file:PFCalibration.db"


process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                  CondDBCommon,
                                  toPut = cms.VPSet(cms.PSet(record = cms.string('PFCalibrationRcd'),
                                                             tag = cms.string('PFCalibration_v10_mc'),
                                                             timetype   = cms.untracked.string('runnumber')
                                                             )
                                                             ),
                                  loadBlobStreamer = cms.untracked.bool(False),
                                  #    timetype   = cms.untracked.string('lumiid')
                                  #    timetype   = cms.untracked.string('runnumber')
                                  )

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = '90X_upgrade2017_realistic_v20'
#process.GlobalTag.connect   = 'sqlite_file:/afs/cern.ch/user/c/cerminar/public/Alca/GlobalTag/GR_R_311_V2.db'
process.GlobalTag.globaltag = '103X_upgrade2018_realistic_v8'

process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("PFCalibrationRcd"),
             tag = cms.string("PFCalibration_v10_mc"),
             connect = cms.string("sqlite_file:PFCalibration.db")
             #connect = cms.untracked.string("sqlite_file:PFCalibration.db")
             )
    )
