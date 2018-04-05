import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("FlatCalib")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("RecoHI.HiEvtPlaneAlgos.HiEvtPlane_cfi")
process.load("RecoHI.HiEvtPlaneAlgos.hiEvtPlaneFlat_cfi")
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
       'root://xrootd.unl.edu//store/user/mnguyen/Hydjet_Quenched_MinBias_5020GeV/HydjetMB_Reco_730_53XBS/a2111270e3580d5672bd373836ad7c8e/hiReco_DIGI_L1_DIGI2RAW_RAW2DIGI_L1Reco_RECO_100_1_gMP.root',
       'root://xrootd.unl.edu//store/user/mnguyen/Hydjet_Quenched_MinBias_5020GeV/HydjetMB_Reco_730_53XBS/a2111270e3580d5672bd373836ad7c8e/hiReco_DIGI_L1_DIGI2RAW_RAW2DIGI_L1Reco_RECO_101_1_dNN.root',
       'root://xrootd.unl.edu//store/user/mnguyen/Hydjet_Quenched_MinBias_5020GeV/HydjetMB_Reco_730_53XBS/a2111270e3580d5672bd373836ad7c8e/hiReco_DIGI_L1_DIGI2RAW_RAW2DIGI_L1Reco_RECO_102_1_wS1.root'
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

#process.CondDBCommon.connect = "sqlite_file:HeavyIonRPRcd_Hydjet_74x_v02_mc.db"
#process.PoolDBESSource = cms.ESSource("PoolDBESSource",
#                                       process.CondDBCommon,
#                                       toGet = cms.VPSet(cms.PSet(record = cms.string('HeavyIonRPRcd'),
#                                                                  tag = cms.string('HeavyIonRPRcd_Hydjet_74x_v02_mc')
#                                                                  )
#                                                         )
#                                      )

process.GlobalTag.toGet.extend([
        cms.PSet(record = cms.string("HeavyIonRPRcd"),
                 tag = cms.string('HeavyIonRPRcd_Hydjet_74x_v02_mc')
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_PAT_000")
                 )
        ])


process.TFileService = cms.Service("TFileService",
    fileName = cms.string("rpflat.root")
)

process.centralityBin.Centrality = cms.InputTag("hiCentrality")
process.centralityBin.centralityVariable = cms.string("HFtowers")
process.centralityBin.nonDefaultGlauberModel = cms.string("HydjetDrum5")

process.dump = cms.EDAnalyzer("EventContentAnalyzer")
process.hiEvtPlane.nonDefaultGlauberModel = cms.string("HydjetDrum5")
process.hiEvtPlaneFlat.nonDefaultGlauberModel = cms.string("HydjetDrum5")

process.hiEvtPlane.loadDB_ = cms.untracked.bool(True)
process.p = cms.Path(process.heavyIon*process.centralityBin*process.hiEvtPlane*process.hiEvtPlaneFlat)



                        

