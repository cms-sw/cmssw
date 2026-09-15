#-----------------------------------------------------------
# This job skims a MC dataset, selecting only products
# that are needed to run the L1 tracking & measure its
# performance. 
#
# It is typically used to create small datasets that
# the git CI runs on to check new code.
#
# Whenever the L1 tracking group switches to a new default
# MC dataset, this skim should be run on ttbar+0PU MC.
# It should be copied somewhere like /eos/user/t/tomalin/
# using cp on lxplus. And a link created to it from 
# somewhere like 
# https://cernbox.cern.ch/index.php/apps/files/?dir=/& .
# N.B. The "quicklink" this gives is buggy. Take the encoded 
# string from it and insert it into something like:
#https://cernbox.cern.ch/remote.php/dav/public-files/4wMLEX986bdIs8U/skimmedForCI_14_0_0.root
# The link to the skimmed dataset should be referred to in
# https://gitlab.cern.ch/cms-l1tk/cmssw_CI .
#-----------------------------------------------------------

############################################################
# define basic process
############################################################

import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
import os
process = cms.Process("SKIM")

 
############################################################
# import standard configurations
############################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

# D110 geometry
process.load('Configuration.Geometry.GeometryExtendedRun4D121Reco_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4D121_cff')

process.load('Configuration.StandardSequences.EndOfProcess_cff')


process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))

#--- To use MCsamples scripts, defining functions get*data*() for easy MC access,
#--- follow instructions in https://github.com/cms-L1TK/MCsamples

#from MCsamples.Scripts.getCMSdata_cfi import *

# Read data from card files (defines getCMSdataFromCards()):
#from MCsamples.RelVal_1510_D110.PU0_TTbar_14TeV_cfi import *
#inputMC = getCMSdataFromCards()

inputMC = ['/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/0a33391d-2eb7-45fe-98fe-ca7b70a785c4.root','/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/265da81c-4169-4fd1-a41d-20945604cf49.root',
'/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/31621cff-67fa-41b5-887d-8f70d62f5494.root','/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/636a41e0-61f7-4e47-961c-68ce96cdd620.root',
'/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/64c6656e-33fb-47e2-a557-8ffa2c01a40b.root','/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/7673b33d-a881-4e61-844c-3025c51b129e.root',
'/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/7d05d892-22d8-4456-a3c9-5cc0cc86bf54.root','/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/7fdba3a5-39e4-4d03-88eb-91403d5a5e39.root',
'/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/a5970f6d-59b6-4d62-bebf-309022803771.root','/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/bc9c3601-7ee1-4a4a-91a4-f19666db9bf0.root',
'/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/ee189f04-faef-45bf-8f48-50a784da30d7.root','/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/f54e766f-da14-43e4-93de-2554ef9b95ff.root']  

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(*inputMC))

process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring('drop *'),
    fileName = cms.untracked.string('/tmp/skimmedForCI.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW')
    )
)

process.output.outputCommands.append('keep  *_*_*Level1TTTracks*_*')
process.output.outputCommands.append('keep  *_*_*StubAccepted*_*')
process.output.outputCommands.append('keep  *_*_*ClusterAccepted*_*')
process.output.outputCommands.append('keep  *_*_*MergedTrackTruth*_*')
process.output.outputCommands.append('keep  *_genParticles_*_*')

# Add this if you need to rereconstruct the stubs.
#process.output.outputCommands.append('keep  Phase2TrackerDigi*_mix_Tracker_*')
#process.output.outputCommands.append('keep  PixelDigiSimLinked*_simSiPixelDigis_Tracker_*')

process.pd = cms.EndPath(process.output)

process.schedule = cms.Schedule(process.pd)
