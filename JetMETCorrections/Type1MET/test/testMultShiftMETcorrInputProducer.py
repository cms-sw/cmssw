import FWCore.ParameterSet.Config as cms
process = cms.Process("test")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
process.source = cms.Source(
    'PoolSource',
# CSA14 RECO
#    fileNames = cms.untracked.vstring('root://eoscms.cern.ch//eos/cms/store/relval/CMSSW_7_0_5/RelValTTbar_13/GEN-SIM-RECO/POSTLS170_V6-v3/00000/0423767B-B5DD-E311-A1E0-02163E00E5B5.root') 
# 720 RECO
    fileNames = cms.untracked.vstring('root://eoscms.cern.ch//store/relval/CMSSW_7_5_0_pre4/RelValTTbar_13/GEN-SIM-RECO/MCRUN2_75_V1-v1/00000/469C34DB-12F6-E411-B012-0025905B855C.root')
# PHYS14 mAOD
#    fileNames = cms.untracked.vstring('root://xrootd.unl.edu//store/mc/Phys14DR/TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola/MINIAODSIM/PU30bx50_PHYS14_25_V1-v1/00000/003B6371-8D81-E411-8467-003048F0E826.root')
# PHYS14 mAOD local
#    fileNames = cms.untracked.vstring('file:/data/schoef/local/TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola_MINIAODSIM_PU20bx25_PHYS14_25_V1-v1.root')
    )


process.load('Configuration.StandardSequences.Services_cff')
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = 'MCRUN2_75_V1'
process.load('JetMETCorrections.Type1MET.correctionTermsPfMetMult_cff')

process.out = cms.OutputModule("PoolOutputModule",
     #verbose = cms.untracked.bool(True),
#     SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
     fileName = cms.untracked.string('histo.root'),
     outputCommands = cms.untracked.vstring('keep *_*_*_test','keep recoPFMETs_*_*_*') 
)
process.load('JetMETCorrections.Type1MET.correctedMet_cff')
process.load('JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff')
#
# RUN!
#
process.run = cms.Path(
  process.correctionTermsPfMetMult
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
