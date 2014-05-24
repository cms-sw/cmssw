import FWCore.ParameterSet.Config as cms
process = cms.Process("test")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring('root://xrootd.unl.edu//store/relval/CMSSW_7_0_5/RelValTTbar_13/GEN-SIM-RECO/POSTLS170_V6-v3/00000/0423767B-B5DD-E311-A1E0-02163E00E5B5.root')
    )


process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'POSTLS170_V6::All'
process.load('JetMETCorrections.Type1MET.correctionTermsPfMetMult_cff')

process.out = cms.OutputModule("PoolOutputModule",
     #verbose = cms.untracked.bool(True),
#     SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
     fileName = cms.untracked.string('histo.root'),
     outputCommands = cms.untracked.vstring('keep *') 
)
process.load('JetMETCorrections.Type1MET.correctedMet_cff')
#
# RUN!
#
process.run = cms.Path(
  process.correctionTermsPfMetMult*
  process.pfMetMultCorr 
)

process.outpath = cms.EndPath(process.out)

