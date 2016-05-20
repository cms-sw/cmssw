import FWCore.ParameterSet.Config as cms
process = cms.Process("test")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 500
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )



# DataBase
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:MET16V0.db'


#process.source = cms.Source("EmptySource")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
      process.CondDB,
      timetype = cms.string('runnumber'),
      toGet = cms.VPSet(
      cms.PSet(
              #record = cms.string('MetShiftXY'),
              #record = cms.string('PfType1Met'), 
              record = cms.string('METCorrectionsRecord'),# plugin 
              #tag    = cms.string('metShiftxy'),
              tag    = cms.string('METCorrectorParametersCollection_MET16V0'),
              #label  = cms.untracked.string('PfType1Met')
              label  = cms.untracked.string('PfType1MetLocal')
              #label  = cms.untracked.string('AK5CaloLocal') 
            )                                                                               
       )
)

process.source = cms.Source(
    'PoolSource',
# CSA14 RECO
#    fileNames = cms.untracked.vstring('root://eoscms.cern.ch//eos/cms/store/relval/CMSSW_7_0_5/RelValTTbar_13/GEN-SIM-RECO/POSTLS170_V6-v3/00000/0423767B-B5DD-E311-A1E0-02163E00E5B5.root') 
# 720 RECO
    #fileNames = cms.untracked.vstring('root://eoscms.cern.ch//store/relval/CMSSW_7_5_0_pre4/RelValTTbar_13/GEN-SIM-RECO/MCRUN2_75_V1-v1/00000/469C34DB-12F6-E411-B012-0025905B855C.root')
# PHYS14 mAOD
#    fileNames = cms.untracked.vstring('root://xrootd.unl.edu//store/mc/Phys14DR/TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola/MINIAODSIM/PU30bx50_PHYS14_25_V1-v1/00000/003B6371-8D81-E411-8467-003048F0E826.root')
# PHYS14 mAOD local
#    fileNames = cms.untracked.vstring('file:/data/schoef/local/TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola_MINIAODSIM_PU20bx25_PHYS14_25_V1-v1.root')
    )


process.load('Configuration.StandardSequences.Services_cff')
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = 'MCRUN2_75_V1'
process.load('JetMETCorrections.Type1MET.correctionTermsPfMetMult_cff')
#process.load('JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff')
process.corrPfMetXYMultDB.payloadName = cms.untracked.string('PfType1MetLocal')
process.corrPfMetXYMultDB.globalTag = cms.untracked.string('MET16V0')

process.out = cms.OutputModule("PoolOutputModule",
     #verbose = cms.untracked.bool(True),
#     SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
     fileName = cms.untracked.string('histo.root'),
     outputCommands = cms.untracked.vstring('keep *_*_*_test','keep recoPFMETs_*_*_*','keep *_slimmedMETs_*_*','keep *_offlineSlimmedPrimaryVertices_*_*') 
)
process.load('JetMETCorrections.Type1MET.correctedMet_cff')
process.pfMetTxy.isMiniAod = cms.bool(True)
process.pfMetTxy.src = cms.InputTag('slimmedMETs')
process.pfMetTxy.srcCorrections = cms.VInputTag('corrPfMetXYMultDB')
#
# RUN!
#
process.run = cms.Path(
  process.correctionTermsPfMetMultDB
#  *process.correctionTermsPfMetType1Type2
#  *process.pfMetT0rt
#  *process.pfMetT0rtT1
#  *process.pfMetT0rtT1T2
#  *process.pfMetT0rtT2
#  *process.pfMetT0pc
#  *process.pfMetT0pcT1
#  *process.pfMetT1
#  *process.pfMetT1T2
  *process.pfMetTxy
#  *process.pfMetT0rtTxy
#  *process.pfMetT0rtT1Txy
#  *process.pfMetT0rtT1T2Txy
#  *process.pfMetT0pcTxy
#  *process.pfMetT0pcT1Txy
#  *process.pfMetT0pcT1T2Txy
#  *process.pfMetT1Txy
#  *process.pfMetT1T2Txy
)

process.outpath = cms.EndPath(process.out)
