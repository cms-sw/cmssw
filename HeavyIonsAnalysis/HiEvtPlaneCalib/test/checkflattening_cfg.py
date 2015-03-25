import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("FlatCalib")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("RecoHI.HiEvtPlaneAlgos.HiEvtPlane_cfi")
process.load("RecoHI.HiEvtPlaneAlgos.hievtplaneflatproducer_cfi")
process.load("HeavyIonsAnalysis.HiEvtPlaneCalib/checkflattening_cfi")
process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load('GeneratorInterface.HiGenCommon.HeavyIon_cff')
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.GlobalTag.globaltag='GR_R_74_V0A::All'
process.MessageLogger.cerr.FwkReport.reportEvery=1000
process.HeavyIonGlobalParameters = cms.PSet(
    centralityVariable = cms.string("HFtowers"),
    centralitySrc = cms.InputTag("hiCentrality")
    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)
process.GlobalTag.toGet.extend([

 cms.PSet(record = cms.string("HeavyIonRcd"),
  tag = cms.string("CentralityTable_HFtowers200_Glauber2010A_v5315x02_offline"),
  connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
  label = cms.untracked.string("HFtowers")
 )
])

#readFiles = cms.untracked.vstring()

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
       'root://xrootd.unl.edu//store/relval/CMSSW_7_4_0_pre6/HIMinBiasUPC/RECO/GR_R_74_V0A_RelVal_hi2011-v1/00000/00726191-ACA8-E411-AA70-02163E00E994.root',
       'root://xrootd.unl.edu//store/relval/CMSSW_7_4_0_pre6/HIMinBiasUPC/RECO/GR_R_74_V0A_RelVal_hi2011-v1/00000/007E7AE0-ABA8-E411-9F36-02163E00E91E.root',
       'root://xrootd.unl.edu//store/relval/CMSSW_7_4_0_pre6/HIMinBiasUPC/RECO/GR_R_74_V0A_RelVal_hi2011-v1/00000/02044B3B-B2A8-E411-B5B5-02163E00CAB8.root'
),
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
                            inputCommands=cms.untracked.vstring(
        'keep *',
        'drop *_hiEvtPlane_*_*'
        ),
                            dropDescendantsOfDroppedBranches=cms.untracked.bool(False)
                            )



#process.CondDBCommon.connect = "sqlite_file:HeavyIonRPRcd_PbPb2011_74X_v01_offline.db"
#process.PoolDBESSource = cms.ESSource("PoolDBESSource",
#                                       process.CondDBCommon,
#                                       toGet = cms.VPSet(cms.PSet(record = cms.string('HeavyIonRPRcd'),
#                                                                  tag = cms.string('HeavyIonRPRcd_PbPb2011_74X_v01_offline')
#                                                                  )
#                                                         )
#                                      )

process.GlobalTag.toGet.extend([
        cms.PSet(record = cms.string("HeavyIonRPRcd"),
                 tag = cms.string('HeavyIonRPRcd_PbPb2011_74X_v01_offline'),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_PAT_000")
                 )
        ])

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("rpflat.root")
)

process.centralityBin.Centrality = cms.InputTag("hiCentrality")
process.centralityBin.centralityVariable = cms.string("HFtowers")
process.centralityBin.nonDefaultGlauberModel = cms.string("")

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.hiEvtPlane.loadDB_ = cms.untracked.bool(True)
process.hiEvtPlaneFlat.genFlatPsi_ = cms.untracked.bool(True)
process.p = cms.Path(process.centralityBin*process.hiEvtPlane*process.hiEvtPlaneFlat*process.checkflattening)


                        

