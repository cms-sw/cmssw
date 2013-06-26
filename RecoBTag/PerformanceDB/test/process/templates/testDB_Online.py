import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.load ("Pool_NAME")
process.load ("Btag_NAME")

#
# change inside the source
#

#BtagPerformanceESProducer_mistagJPL = cms.ESProducer("BtagPerformanceESProducer",
#                                                     # this is what it makes available
#                                                     ComponentName = cms.string('Mistag_JPL'),
#                                                     # this is where it gets the payload from                                                
#                                                     PayloadName = cms.string('MISTAGJPL_T'),
#                                                     WorkingPointName = cms.string('MISTAGJPL_WP')
#                                                     )
#BtagPerformanceESProducer_mistagJPM = cms.ESProducer("BtagPerformanceESProducer",
#                                                     # this is what it makes available
#                                                     ComponentName = cms.string('Mistag_JPM'),
#                                                     # this is where it gets the payload from                                                
#                                                     PayloadName = cms.string('MISTAGJPM_T'),
#                                                     WorkingPointName = cms.string('MISTAGJPM_WP')
#                                                     )
#BtagPerformanceESProducer_mistagJPT = cms.ESProducer("BtagPerformanceESProducer",
#                                                     # this is what it makes available
#                                                     ComponentName = cms.string('Mistag_JPT'),
#                                                     # this is where it gets the payload from                                                
#                                                     PayloadName = cms.string('MISTAGJPT_T'),
#                                                     WorkingPointName = cms.string('MISTAGJPT_WP')
#                                                     )
#



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.source = cms.Source("EmptySource")




process.demo = cms.EDAnalyzer('validateBTagDB',
                              CalibrationForBEfficiency = cms.string('MISTAGSSVM'),
                              CalibrationForCEfficiency = cms.string('MISTAGSSVM'),
                              CalibrationForMistag = cms.string('MISTAGSSVM'),
                              algoNames = cms.vstring("NAME"),                              
                              fileList = cms.vstring("../FILE"),                              
                              )

process.p = cms.Path(process.demo)

#

