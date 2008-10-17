import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("EmptySource",
)
process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.demo = cms.EDAnalyzer('SiPixelDQMRocLevelAnalyzer',

                              fileName = cms.untracked.string("../../../../../../../SHIFT/071008/rootfiles64656/DQM_PixelAlive_Run64656.root"),

                              barrelRocStud = cms.untracked.bool(True),
                              
                              endcapRocStud = cms.untracked.bool(True),

                              pixelAliveStudy = cms.untracked.bool(True),
                              pixelAliveThreshold = cms.untracked.double(0.),

                              thresholdStudy = cms.untracked.bool(False),

                              noiseStudy = cms.untracked.bool(False),

                              gainStudy = cms.untracked.bool(False),

                              pedestalStudy = cms.untracked.bool(False)
)

process.TFileService = cms.Service("TFileService",
                                  fileName = cms.string("histos.root")
                                  )


process.p = cms.Path(process.demo)
