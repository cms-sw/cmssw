import FWCore.ParameterSet.Config as cms
import sys 
import os 

process = cms.Process("PFAOD")


# from CMGTools.Production.datasetToSource import *
# process.source = datasetToSource(
#     'CMS',
#     '/DoubleMu/StoreResults-DoubleMu_2011A_PR_v4_embedded_trans1_tau116_ptmu1_13had1_17_v3-f456bdbb960236e5c696adfe9b04eaae/USER',
#     '.*root')

process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring(
                                 '/store/cmst3/user/botta/PFAODContentStudy/AOD_DoubleMu_muIsoDeposit.root',
 #                                '/store/cmst3/user/botta/PFAODContentStudy/AOD_DoubleMu_V3.root',
 #                                 '/store/cmst3/user/botta/PFAODContentStudy/AOD_DoubleElectron_muIsoDeposit2.root',
 
                                 ))


process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False))
#WARNING!
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("Configuration.EventContent.EventContent_cff")
process.out = cms.OutputModule(
    "PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string( 'PFAOD.root' ),
    )

from CMGTools.Production.PFAOD.PFAOD_EventContent_cff import V5
process.out.outputCommands.extend( V5 )

process.endpath = cms.EndPath(
    process.out
    )


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100


