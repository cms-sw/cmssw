import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *

geometryESSource = cms.ESSource("PoolDBESSource",
                                 CondDBSetup,
                                    toGet = cms.VPSet(cms.PSet(record = cms.string('GlobalPositionRcd'),
                                                               tag = cms.string('IdealGeometry')
                                                               ),
                                                      cms.PSet(record = cms.string('DTAlignmentRcd'),
                                                               tag = cms.string('DTIdealGeometry200_mc')
                                                               ),
                                                      cms.PSet(record = cms.string('DTAlignmentErrorRcd'),
                                                               tag = cms.string('DTIdealGeometryErrors200_mc')
                                                               ),
                                                      cms.PSet(record = cms.string('CSCAlignmentRcd'),
                                                               tag = cms.string('CSCIdealGeometry200_mc')
                                                               ),
                                                      cms.PSet(record = cms.string('CSCAlignmentErrorRcd'),
                                                               tag = cms.string('CSCIdealGeometryErrors200_mc')
                                                               ),
                                                      cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                                                               tag = cms.string('TrackerIdealGeometry210_mc')
                                                               ),
                                                      cms.PSet(record = cms.string('TrackerAlignmentErrorRcd'),
                                                               tag = cms.string('TrackerIdealGeometryErrors210_mc')
                                                                                                                                                                                                                        )
                                                      ),
                                    connect = cms.string('frontier://cms_conditions_data/CMS_COND_21X_ALIGNMENT')
                                    )

beamSpotESSource = cms.ESSource("PoolDBESSource",
                              CondDBSetup,
                              timetype = cms.string('runnumber'),
                              toGet = cms.VPSet(cms.PSet(record = cms.string('BeamSpotObjectsRcd'),
                                                         tag = cms.string('Early10TeVCollision_3p8cm_v2_mc')
                                                         )
                                                ),
                              connect = cms.string('frontier://Frontier/CMS_COND_21X_BEAMSPOT'),
                              authenticationMethod = cms.untracked.uint32(0)
                              )

dtConditionsESSource = cms.ESSource("PoolDBESSource",
                         CondDBSetup,
                         timetype = cms.string('runnumber'),
                         toGet = cms.VPSet(cms.PSet(record = cms.string('DTReadOutMappingRcd'),
                                                    tag = cms.string('DT_map10ddu21X_V01')
                                                    ),
                                           cms.PSet(record = cms.string('DTT0Rcd'),
                                                    tag = cms.string('t0_CRUZET_hlt')
                                                    ),
                                           cms.PSet(record = cms.string('DTStatusFlagRcd'),
                                                    tag = cms.string('noise_CRUZET_hlt')
                                                    ),
                                           cms.PSet(record = cms.string('DTMtimeRcd'),
                                                    tag = cms.string('vDrift_LHCStartUp_543_CMSSW219')
                                                    )
                                           ),
                         connect = cms.string('frontier://Frontier/CMS_COND_21X_DT'),
                         authenticationMethod = cms.untracked.uint32(0)
                         )

# process.es_prefer_roMapping = cms.ESPrefer('PoolDBESSource','roMapping')
ttrigsource = cms.ESSource("PoolDBESSource",
                           CondDBSetup,
                           timetype = cms.string('runnumber'),
                           toGet = cms.VPSet(cms.PSet(record = cms.string('DTTtrigRcd'),
                                                      tag = cms.string('tTrig_CRAFT_081021_1614_offline')
                                                      )
                                             ),
                           connect = cms.string('frontier://Frontier/CMS_COND_21X_DT'),
                           authenticationMethod = cms.untracked.uint32(0)
                           )
