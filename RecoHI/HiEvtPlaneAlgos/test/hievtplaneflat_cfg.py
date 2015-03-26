import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("FlatCalib")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("RecoHI.HiEvtPlaneAlgos.HiEvtPlane_cfi")
process.load("RecoHI.HiEvtPlaneAlgos.hievtplaneflatproducer_cfi")
process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load('GeneratorInterface.HiGenCommon.HeavyIon_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc_HIon', '')

process.HeavyIonGlobalParameters = cms.PSet(
    centralityVariable = cms.string("HFtowers"),
    centralitySrc = cms.InputTag("hiCentrality")
    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.FwkReport.reportEvery=1000

process.source = cms.Source("PoolSource",
#                            fileNames = readFiles, secondaryFileNames = secFiles,
    fileNames = cms.untracked.vstring(
#       '/store/user/mnguyen/Hydjet_Quenched_MinBias_5020GeV/HydjetMB_740pre8_MCHI2_74_V3_53XBS_RECO_v5/fa4d7cedb51d6196cc0424fd90debe3f/step3_RAW2DIGI_L1Reco_RECO_100_1_FN3.root',
#       '/store/user/mnguyen/Hydjet_Quenched_MinBias_5020GeV/HydjetMB_740pre8_MCHI2_74_V3_53XBS_RECO_v5/fa4d7cedb51d6196cc0424fd90debe3f/step3_RAW2DIGI_L1Reco_RECO_101_1_1dG.root',
        ),
                            inputCommands=cms.untracked.vstring(
        'keep *',
        'drop *_hiEvtPlane_*_*'
        )

)
process.source.duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

process.GlobalTag.toGet.extend([
   cms.PSet(record = cms.string("HeavyIonRcd"),
      tag = cms.string("CentralityTable_HFtowers200_HydjetDrum5_v740x01_mc"),
      connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
      label = cms.untracked.string("HFtowersHydjetDrum5")
   ),
])

process.CondDBCommon.connect = "sqlite_file:HeavyIonRPRcd_Hydjet_74x_v02_mc.db"
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                       process.CondDBCommon,
                                       toGet = cms.VPSet(cms.PSet(record = cms.string('HeavyIonRPRcd'),
                                                                  tag = cms.string('HeavyIonRPRcd_Hydjet_74x_v02_mc')
                                                                  )
                                                         )
                                      )

#process.GlobalTag.toGet.extend([
#        cms.PSet(record = cms.string("HeavyIonRPRcd"),
#                 tag = cms.string('HeavyIonRPRcd_Hydjet_74x_v01_mc'),
#                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_PAT_000")
#                 )
#        ])


process.TFileService = cms.Service("TFileService",
    fileName = cms.string("rpflat.root")
)

process.centralityBin.Centrality = cms.InputTag("hiCentrality")
process.centralityBin.centralityVariable = cms.string("HFtowers")
process.centralityBin.nonDefaultGlauberModel = cms.string("HydjetDrum5")

process.dump = cms.EDAnalyzer("EventContentAnalyzer")
process.hiEvtPlane.nonDefaultGlauberModel = cms.string("HydjetDrum5")
process.hiEvtPlaneFlatCalib.nonDefaultGlauberModel = cms.string("HydjetDrum5")

process.hiEvtPlane.loadDB_ = cms.untracked.bool(True)
process.hiEvtPlaneFlatCalib.genFlatPsi_ = cms.untracked.bool(True)
process.p = cms.Path(process.heavyIon*process.centralityBin*process.hiEvtPlane*process.hiEvtPlaneFlatCalib)



                        

