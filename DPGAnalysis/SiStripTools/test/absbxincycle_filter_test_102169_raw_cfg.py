import FWCore.ParameterSet.Config as cms

process = cms.Process("RAWFilterTest")

process.load("DPGAnalysis.SiStripTools.filtertest.config_102169_raw_cff")
#------------------------------------------------------------------
# filters
#------------------------------------------------------------------
process.load("DPGAnalysis.SiStripTools.filtertest.absbxincycle_filter_tests_cff")
#------------------------------------------------------------------


process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('absbxincycle_filter_test_102169_raw.root')
                                   )

process.p0 = cms.Path(process.sinit + process.eventtimedistribution)

process.p1 = cms.Path(process.sinit + process.absbxincycles1)
process.p2 = cms.Path(process.sinit + process.absbxincycles2)
process.p3 = cms.Path(process.sinit + process.absbxincycles3)
process.p4 = cms.Path(process.sinit + process.absbxincycles4)
process.p5 = cms.Path(process.sinit + process.absbxincycles5)
process.p6 = cms.Path(process.sinit + process.absbxincycles6)

process.p11 = cms.Path(process.sinit + process.absbxincycles11)
process.p12 = cms.Path(process.sinit + process.absbxincycles12)
process.p13 = cms.Path(process.sinit + process.absbxincycles13)
process.p14 = cms.Path(process.sinit + process.absbxincycles14)
process.p15 = cms.Path(process.sinit + process.absbxincycles15)
process.p16 = cms.Path(process.sinit + process.absbxincycles16)

process.p21 = cms.Path(process.sinit + process.absbxincycles21)
process.p22 = cms.Path(process.sinit + process.absbxincycles22)

process.p31 = cms.Path(process.sinit + process.absbxincycles31)
