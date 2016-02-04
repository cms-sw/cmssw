import FWCore.ParameterSet.Config as cms

process = cms.Process("ANA")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("GeneratorInterface.HydjetInterface.hydjetDefault_cfi")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10)
                                       )

process.SimpleMemoryCheck = cms.Service('SimpleMemoryCheck',
                                        ignoreTotal=cms.untracked.int32(0),
                                        oncePerEventMode = cms.untracked.bool(False)
                                        )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
#'/store/user/davidlw/AMPT_B0_2760GeV_GEN/amptDefault_cfi_py_GEN_1.root'
'/store/user/davidlw/AMPT_B0_2760GeV_GEN/amptDefault_cfi_py_GEN_2.root'
#,'/store/user/davidlw/AMPT_B0_2760GeV_GEN/amptDefault_cfi_py_GEN_3.root'
#,'/store/user/davidlw/AMPT_B0_2760GeV_GEN/amptDefault_cfi_py_GEN_4.root'
,'/store/user/davidlw/AMPT_B0_2760GeV_GEN/amptDefault_cfi_py_GEN_5.root'
#,'/store/user/davidlw/AMPT_B0_2760GeV_GEN/amptDefault_cfi_py_GEN_6.root'
#,'/store/user/davidlw/AMPT_B0_2760GeV_GEN/amptDefault_cfi_py_GEN_7.root'
#,'/store/user/davidlw/AMPT_B0_2760GeV_GEN/amptDefault_cfi_py_GEN_8.root'
,'/store/user/davidlw/AMPT_B0_2760GeV_GEN/amptDefault_cfi_py_GEN_9.root'
#,'/store/user/davidlw/AMPT_B0_2760GeV_GEN/amptDefault_cfi_py_GEN_10.root'
#'/store/user/davidlw/Hydjet_Quenched_MinBias_2760GeV_MC_3XY_V24_GEN/Hydjet_Quenched_MinBias_2760GeV_MC_3XY_V24_GEN/3adac1026f47b564bed65f2efae39879/Hydjet_Quenched_MinBias_2760GeV_cfi_py_GEN_537.root'
#'file:amptDefault_cfi_py_GEN.root'
)
                            ) 

process.ana = cms.EDAnalyzer('AMPTAnalyzer'
                             )

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('treefile_ampt.root')
                                   )
process.p = cms.Path(process.ana)




