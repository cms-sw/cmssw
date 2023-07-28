#-----------------------------------------------------------
# This job skims a MC dataset, selecting only products
# that are needed to run the L1 tracking & measure its
# performance. 
#
# It is typically used to create small datasets that
# the git CI runs on to check new code.
#
# Whenever the L1 tracking group switches to a new default
# MC dataset, this skim should be run on ttbar+0PU MC, 
# and the skimmed dataset placed in
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

# D88 geometry (T24 tracker)
process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D88_cff')

process.load('Configuration.StandardSequences.EndOfProcess_cff')


process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))

#--- To use MCsamples scripts, defining functions get*data*() for easy MC access,
#--- follow instructions in https://github.com/cms-L1TK/MCsamples

from MCsamples.Scripts.getCMSdata_cfi import *

# Read data from card files (defines getCMSdataFromCards()):
#from MCsamples.RelVal_1130_D76.PU0_TTbar_14TeV_cfi import *
#inputMC = getCMSdataFromCards()

dataName="/RelValTTbar_14TeV/CMSSW_12_6_0_pre4-125X_mcRun4_realistic_v2_2026D88noPU-v1/GEN-SIM-DIGI-RAW"
inputMC=getCMSdata(dataName)

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

process.pd = cms.EndPath(process.output)

process.schedule = cms.Schedule(process.pd)
