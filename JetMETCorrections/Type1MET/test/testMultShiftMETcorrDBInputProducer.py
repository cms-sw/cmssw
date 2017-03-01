import FWCore.ParameterSet.Config as cms
process = cms.Process("test")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 500
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

# DataBase
process.load("CondCore.CondDB.CondDB_cfi")
#process.CondDB.connect = 'sqlite_file:../data/Summer16_V0_DATA_MEtXY.db'
process.CondDB.connect = 'sqlite_file:../data/Summer16_V0_MC_MEtXY.db'


#process.source = cms.Source("EmptySource")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
      process.CondDB,
      timetype = cms.string('runnumber'),
      toGet = cms.VPSet(
      cms.PSet(
              record = cms.string('MEtXYcorrectRecord'),# plugin 
              #tag    = cms.string('MEtXYcorrectParametersCollection_Summer16_V0_DATA_PfType1Met'), 
              tag    = cms.string('MEtXYcorrectParametersCollection_Summer16_V0_MC_PfType1Met'), 
              #label  = cms.untracked.string('PfType1Met')
              label  = cms.untracked.string('PfType1MetLocal')
            )                                                                               
       )
)

process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring('file:/u/user/salee/Data/ReValZMM_13/CMSSW_8_1_0_pre6-PU25ns_80X_mcRun2_asymptotic_v14-v1/MINIAODSIM/A0DE71C7-D82C-E611-88FF-0025905B85AA.root')
# 720 RECO
    #fileNames = cms.untracked.vstring('root://eoscms.cern.ch//store/relval/CMSSW_7_5_0_pre4/RelValTTbar_13/GEN-SIM-RECO/MCRUN2_75_V1-v1/00000/469C34DB-12F6-E411-B012-0025905B855C.root')
# PHYS14 mAOD
#    fileNames = cms.untracked.vstring('root://xrootd.unl.edu//store/mc/Phys14DR/TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola/MINIAODSIM/PU30bx50_PHYS14_25_V1-v1/00000/003B6371-8D81-E411-8467-003048F0E826.root')
# PHYS14 mAOD local
    )


process.load('Configuration.StandardSequences.Services_cff')
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = 'MCRUN2_75_V1'
process.load('JetMETCorrections.Type1MET.correctionTermsPfMetMultDB_cff')
process.corrPfMetXYMultDB.payloadName = cms.string('PfType1MetLocal')
#process.corrPfMetXYMultDB.isData = cms.bool(True)
process.corrPfMetXYMultDB.isData = cms.bool(False)
#process.corrPfMetXYMultDB.sampleType = cms.untracked.string('MC') #DY, MC (default), TTJets, WJets, Data

process.load('JetMETCorrections.Type1MET.correctedPatMet_cff')
#process.pfMetTxy.isMiniAod = cms.bool(True) default
#process.patMetTxy.src = cms.InputTag('slimmedMETs') default
#process.patMetTxy.srcCorrections = cms.VInputTag('corrPfMetXYMultDB') default

process.out = cms.OutputModule("PoolOutputModule",
     #verbose = cms.untracked.bool(True),
#     SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
     fileName = cms.untracked.string('histo.root'),
     outputCommands = cms.untracked.vstring('keep *_*_*_test','keep recoPFMETs_*_*_*','keep *_slimmedMETs_*_*','keep *_offlineSlimmedPrimaryVertices_*_*') 
)
#
# RUN!
#
process.run = cms.Path(
  process.correctionTermsPfMetMultDB
  *process.patMetTxy
)

process.outpath = cms.EndPath(process.out)
