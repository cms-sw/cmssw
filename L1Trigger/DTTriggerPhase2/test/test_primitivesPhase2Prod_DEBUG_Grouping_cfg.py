import FWCore.ParameterSet.Config as cms

process = cms.Process("L1DTTrigPhase2Prod")

process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')

process.load("L1Trigger.DTTriggerPhase2.dtTriggerPhase2PrimitiveDigis_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# process.GlobalTag.globaltag = "80X_dataRun2_2016SeptRepro_v7"
# process.GlobalTag.globaltag = "106X_upgrade2023_realistic_v3"

process.load("L1Trigger.DTTriggerPhase2.CalibratedDigis_cfi")
process.load("L1Trigger.DTTriggerPhase2.dtTriggerPhase2PrimitiveDigis_cfi")

process.dtTriggerPhase2PrimitiveDigis.dump = True
process.dtTriggerPhase2PrimitiveDigis.debug = True
# process.dtTriggerPhase2PrimitiveDigis.chi2Th = cms.untracked.double(0.16)

#process.load("FWCore.MessageLogger.MessageLogger_cfi")
#
#process.MessageLogger = cms.Service("MessageLogger",
#                                    destinations = cms.untracked.vstring("detailedInfo"),
#                                    detailedInfo = cms.untracked.PSet(threshold = cms.untracked.string("DEBUG"),
#                                                                      default = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
#                                                                      extension = cms.untracked.string(".txt")),
#                                    debugModules = cms.untracked.vstring("dtTriggerPhase2BayesPrimitiveDigis","DTTrigPhase2Prod"),
#                                )

#scenario
process.dtTriggerPhase2PrimitiveDigis.scenario = 0 #0 is mc, 1 is data, 2 is slice test
process.CalibratedDigis.dtDigiTag = "simMuonDTDigis"
process.CalibratedDigis.scenario = 0

# Bayes
process.dtTriggerPhase2BayesPrimitiveDigis = process.dtTriggerPhase2PrimitiveDigis.clone()
process.dtTriggerPhase2BayesPrimitiveDigis.algo = 1 ## bayes grouping

process.dtTriggerPhase2BayesPrimitiveDigis.minHits4Fit = 3
process.dtTriggerPhase2BayesPrimitiveDigis.df_extended = 2
process.dtTriggerPhase2BayesPrimitiveDigis.PseudoBayesPattern.minNLayerHits = 3
process.dtTriggerPhase2BayesPrimitiveDigis.PseudoBayesPattern.minSingleSLHitsMax = 3 
process.dtTriggerPhase2BayesPrimitiveDigis.PseudoBayesPattern.minSingleSLHitsMin = 0 
process.dtTriggerPhase2BayesPrimitiveDigis.PseudoBayesPattern.allowedVariance = 1
#process.dtTriggerPhase2BayesPrimitiveDigis.timeTolerance = cms.int32(999999)


# STD
process.dtTriggerPhase2StdPrimitiveDigis   = process.dtTriggerPhase2PrimitiveDigis.clone()
process.dtTriggerPhase2StdPrimitiveDigis.algo = 0 ## initial grouping
process.dtTriggerPhase2StdPrimitiveDigis.df_extended = 2

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                                'file:////eos/cms/store/user/folguera/P2L1TUpgrade/Mu_FlatPt2to100-pythia8-gun_file.root',
                                # '/store/mc/PhaseIITDRSpring19DR/Mu_FlatPt2to100-pythia8-gun/GEN-SIM-DIGI-RAW/PU200_106X_upgrade2023_realistic_v3-v2/70000/941C1EA3-141B-6841-AE07-8E5D3ED57461.root',
                                # '/store/mc/PhaseIITDRSpring19DR/Mu_FlatPt2to100-pythia8-gun/GEN-SIM-DIGI-RAW/PU200_106X_upgrade2023_realistic_v3-v2/70000/A37FFE18-21EF-5648-AFC8-56BF9CA76B58.root',
                                # '/store/mc/PhaseIITDRSpring19DR/Mu_FlatPt2to100-pythia8-gun/GEN-SIM-DIGI-RAW/PU200_106X_upgrade2023_realistic_v3-v2/70000/86F62E38-D278-2841-8BB4-B25FCD44BFF7.root',
                                # '/store/mc/PhaseIITDRSpring19DR/Mu_FlatPt2to100-pythia8-gun/GEN-SIM-DIGI-RAW/PU200_106X_upgrade2023_realistic_v3-v2/70000/7CA99B54-AA55-5047-8FCB-8E85DA5B85CE.root',
                                # '/store/mc/PhaseIITDRSpring19DR/Mu_FlatPt2to100-pythia8-gun/GEN-SIM-DIGI-RAW/PU200_106X_upgrade2023_realistic_v3-v2/70000/763C6CDF-FF14-A745-AB9E-2A56819BEC78.root',
                                # '/store/mc/PhaseIITDRSpring19DR/Mu_FlatPt2to100-pythia8-gun/GEN-SIM-DIGI-RAW/PU200_106X_upgrade2023_realistic_v3-v2/70000/2E88F5EC-8039-4A4D-A48E-A29E2A8C41E7.root',
                                # '/store/mc/PhaseIITDRSpring19DR/Mu_FlatPt2to100-pythia8-gun/GEN-SIM-DIGI-RAW/PU200_106X_upgrade2023_realistic_v3-v2/70000/FFB599F9-D497-6943-9287-58E740519498.root',
                                # '/store/mc/PhaseIITDRSpring19DR/Mu_FlatPt2to100-pythia8-gun/GEN-SIM-DIGI-RAW/PU200_106X_upgrade2023_realistic_v3-v2/70000/51EFC25B-E0FB-9B4B-B85C-EE4C5CE9C378.root',
                                # '/store/mc/PhaseIITDRSpring19DR/Mu_FlatPt2to100-pythia8-gun/GEN-SIM-DIGI-RAW/PU200_106X_upgrade2023_realistic_v3-v2/70000/F98EB821-7883-F34C-93AD-2CA3E790BC94.root',

)
                        )
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))


####################### SliceTest specials ##############################



process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring(
                                   'drop *',
                                   'keep *_CalibratedDigis_*_*',
                                   'keep *_dtTriggerPhase2BayesPrimitiveDigis_*_*',
                                   'keep *_dtTriggerPhase2StdPrimitiveDigis_*_*',
                                   'keep *_genParticles_*_*',
                               ),
                               fileName = cms.untracked.string('DTTriggerPhase2Primitives.root')
)

process.p = cms.Path(process.CalibratedDigis *
                     process.dtTriggerPhase2BayesPrimitiveDigis *
                     process.dtTriggerPhase2StdPrimitiveDigis
)
process.this_is_the_end = cms.EndPath(process.out)
