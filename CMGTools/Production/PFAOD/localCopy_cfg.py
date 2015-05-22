import FWCore.ParameterSet.Config as cms

process = cms.Process("AOD")

# name =  '/store/data/Run2011B/DoubleMu/AOD/PromptReco-v1/000/175/886/C461BAAD-4EDC-E011-B69B-BCAEC53296FD.root'
# name = '/store/data/Run2011B/DoubleElectron/AOD/PromptReco-v1/000/175/877/7A5D134E-43DD-E011-B149-003048D2C0F0.root'
# name = '/store/data/Run2011B/HT/AOD/PromptReco-v1/000/175/834/E0B1CFCC-B8DB-E011-9062-BCAEC518FF7C.root'
# name = '/store/mc/Fall11/TTJets_TuneZ2_7TeV-madgraph-tauola/AODSIM/PU_S6_START42_V14B-v1/0000/0ACA8BC5-11FA-E011-B119-0018F3D09692.root'

#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring(
#    # DoubleMu
#    name 
#    )
#)

# name = '/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/Summer12-PU_S7_START52_V5-v2/AODSIM'
# name = '/RelValProdTTbar/CMSSW_5_2_3-START52_V5-v1/AODSIM'
name = '/RelValProdQCD_Pt_3000_3500/CMSSW_5_2_3-START52_V5-v1/AODSIM'



from CMGTools.Production.datasetToSource import *
process.source = datasetToSource(
   'CMS',
   name,
   # '/TTJets_TuneZ2star_8TeV-madgraph-tauola/Summer12-PU_S7_START52_V5-v1/AODSIM',
   # 'CMS',
   # '/DoubleMu/Run2012A-PromptReco-v1/AOD'
   )


process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False))
#WARNING!
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2000) )

process.load("Configuration.EventContent.EventContent_cff")
process.out = cms.OutputModule(
    "PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('AOD_%s.root' % ( name.split('/')[1].split('_')[0] )),
    )

print process.out.fileName

process.load("CommonTools.ParticleFlow.PF2PAT_EventContent_cff")
process.out.outputCommands.extend( process.prunedAODForPF2PATEventContent.outputCommands )

# additional stuff for Maxime: 
process.out.outputCommands = ['keep *']

process.endpath = cms.EndPath(
    process.out
    )


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10


