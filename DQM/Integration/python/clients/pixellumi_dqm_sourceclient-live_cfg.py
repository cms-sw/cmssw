from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("PixelLumiDQM")

unitTest=False
if 'unitTest=True' in sys.argv:
    unitTest=True

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siPixelDigis', 
					 'sipixelEDAClient'),
    cout = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')),
    destinations = cms.untracked.vstring('cout')
)

#----------------------------
# Event Source
#-----------------------------

if unitTest:
    process.load("DQM.Integration.config.unittestinputsource_cfi")
    from DQM.Integration.config.unittestinputsource_cfi import options
else:
    # for live online DQM in P5
    process.load("DQM.Integration.config.inputsource_cfi")
    from DQM.Integration.config.inputsource_cfi import options

# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")
#from DQM.Integration.config.fileinputsource_cfi import options

##
#----------------------------
# DQM Environment
#-----------------------------
process.load("DQMServices.Components.DQMEnvironment_cfi")

#----------------------------
# DQM Live Environment
#-----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "PixelLumi"
process.dqmSaver.tag = "PixelLumi"
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = "PixelLumi"
process.dqmSaverPB.runNumber = options.runNumber

if not unitTest:
    process.source.SelectEvents = cms.untracked.vstring("HLT_ZeroBias*","HLT_L1AlwaysTrue*", "HLT_PAZeroBias*", "HLT_PAL1AlwaysTrue*")
#if (process.runType.getRunType() == process.runType.hi_run):

if (process.runType.getRunType() == process.runType.cosmic_run and not unitTest):
    process.source.SelectEvents = ['HLT*SingleMu*']

#----------------------------
# Magnetic Field
#-----------------------------
# 3.8T field
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------------------
# GLOBALTAG
#-------------------------------------------------
# Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Condition for lxplus: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')

#-----------------------
#  Reconstruction Modules
#-----------------------
# Real data raw to digi
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.cpu.IncludeErrors = True

# Local Reconstruction
process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")

#----------------------------------
# High Pileup Configuration Changes
#----------------------------------
#if (process.runType.getRunType() == process.runType.hpu_run):
#    process.DQMEventStreamHttpReader.SelectEvents = cms.untracked.PSet(
#        SelectEvents = cms.vstring('HLT_600Tower*','HLT_L1*','HLT_Jet*','HLT_*Cosmic*','HLT_HT*','HLT_MinBias_*','HLT_Physics*', 'HLT_ZeroBias*','HLT_HcalNZS*'))


process.siPixelDigis.cpu.InputLabel = cms.InputTag("rawDataCollector")
#--------------------------------
# Heavy Ion Configuration Changes
#--------------------------------
if (process.runType.getRunType() == process.runType.hi_run):
    process.load('Configuration.StandardSequences.RawToDigi_Repacked_cff')
    process.siPixelDigis.cpu.InputLabel = "rawDataRepacker"

    if not unitTest:
        process.source.SelectEvents = ['HLT_HIL1MinimumBiasHF2AND*']


#    process.DQMEventStreamHttpReader.SelectEvents = cms.untracked.PSet(
#        SelectEvents = cms.vstring('HLT_HI*'))

#--------------------------
# Pixel DQM Source and Client
#----------------------
process.load("DQM.PixelLumi.PixelLumiDQM_cfi") 

if process.dqmRunConfig.type.value() is "playback":
    process.pixel_lumi_dqm.logFileName = "pixel_lumi.txt"
else:
    process.pixel_lumi_dqm.logFileName = "/nfshome0/dqmpro/pixel_lumi.txt"

print(process.pixel_lumi_dqm.logFileName)
    
#--------------------------
# Service
#--------------------------
process.AdaptorConfig = cms.Service("AdaptorConfig")

#--------------------------
# Filters
#--------------------------

#--------------------------
# Scheduling
#--------------------------
process.Reco = cms.Sequence(process.siPixelDigis*process.siPixelClusters)
process.DQMmodules = cms.Sequence(process.dqmEnv*
  process.pixel_lumi_dqm*
  process.dqmSaver*
  process.dqmSaverPB)

process.p = cms.Path(process.Reco*process.DQMmodules)

### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
