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

# D88 geometry (T24 tracker)
process.load('Configuration.Geometry.GeometryExtendedRun4D98Reco_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4D98_cff')

process.load('Configuration.StandardSequences.EndOfProcess_cff')


process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))

#--- To use MCsamples scripts, defining functions get*data*() for easy MC access,
#--- follow instructions in https://github.com/cms-L1TK/MCsamples

from MCsamples.Scripts.getCMSdata_cfi import *

# Read data from card files (defines getCMSdataFromCards()):
from MCsamples.RelVal_1400_D98.PU0_TTbar_14TeV_cfi import *
inputMC = getCMSdataFromCards()

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
