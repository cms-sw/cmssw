# PYTHON configuration file.
# Description:  Example of Filtering events in a range of Monte Carlo ptHat.
# Author: R. Harris
# Date:  28 - October - 2008
import FWCore.ParameterSet.Config as cms

process = cms.Process("Filter")
process.load("FWCore.MessageService.MessageLogger_cfi")
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'file:/uscms_data/d1/rharris/CMSSW_2_1_8/src/Configuration/GenProduction/test/PYTHIA6_QCDpt_0_15_10TeV_GEN_100Kevts.root')
)
#############   pt_hat Filter  #####
process.filter = cms.EDFilter("ptHatFilter",
    ptHatLowerCut     = cms.double(0.0),
    ptHatUpperCut     = cms.double(15.0) 
)
################### Output definition #########################3
process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_sisCone5GenJets_*_*',
        'keep *_sisCone7GenJets_*_*',
        'keep *_iterativeCone5GenJets_*_*',
        'keep *_genEventScale_*_*',
        'keep *_genParticles_*_*',
	'keep edmGenInfoProduct_*_*_*',),
    fileName = cms.untracked.string('/uscms_data/d1/rharris/CMSSW_2_1_8/src/Configuration/GenProduction/test/PYTHIA6_QCDpt_0_15_10TeV_GEN_100Kevts_ptHatFiltered.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p1')
    ),
)
#############   Path       ###########################
process.p1 = cms.Path(process.filter)
process.outpath = cms.EndPath(process.output)
process.schedule = cms.Schedule(process.p1,process.outpath)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

