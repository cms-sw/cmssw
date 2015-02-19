######################################################################
## File: pixellumidqm_live_cfg.py
##
## Unpacks AlCaLumiPixels, then runs some basic selection,
## followed by the PixelLumi Calculation, separately for the three trigger types.
## NB: only one, ZeroBias trigger, used for now.
## Bug Adder: Amita Raval
######################################################################
import FWCore.ParameterSet.Config as cms
import os

#####################
# Configurable pieces
#####################

process = cms.Process("PixelLumiDQM")

#-------------------------------
# Import standard configurations
#-------------------------------
process.load('Configuration.StandardSequences.Services_cff')
#process.load('Configuration.EventContent.EventContent_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
#process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')

#---------------
# Magnetic Field
#---------------
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
#process.load('Configuration.StandardSequences.MagneticField_38T_cff')

#---------
# GEOMETRY
#---------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#----------
# GLOBALTAG
#----------
# Condition for P5 cluster
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
# Condition for lxplus
#process.load("DQM.Integration.test.FrontierCondition_GT_Offline_cfi") 

#----------------
# DQM Environment
#----------------
process.load("DQMServices.Components.DQMEnvironment_cfi")

#-----------------------------------------------------------------------------
# DQM Live Environment (for integration, replace with standard for deployment)
#-----------------------------------------------------------------------------
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder    = "PixelLumi"
process.dqmSaver.dirName = '.'
#------------------------
#  Reconstruction Modules
#------------------------
# Replace the whole reco config with just the local pixel reco
#-------------------------------------------------------------
#process.load('Configuration.StandardSequences.Reconstruction_cff')
#process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")
#process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
#process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")
process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")

#--------------------------------
# Data - raw to digi; just pixels
#--------------------------------
#process.load("Configuration.StandardSequences.RawToDigi_cff")
#process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
#process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.InputLabel = 'source'
process.siPixelDigis.IncludeErrors = True

#--------
# Filters
#--------
# HLT Filters: trigger path filters for 3 triggers in the AlCaLumi stream
# NB: Only ZeroBiasSelector_cfi being used for now
process.load("DQM.PixelLumi.ZeroBiasSelector_cfi")
process.load("DQM.PixelLumi.AlCaLumiPixelsZeroBiasSelector_cfi")
process.load("DQM.PixelLumi.RandomTriggerSelector_cfi")

# Make three instances of pixel_lumi_dqm; only 1 being used for now
#------------------------------------------------------------------
from DQM.PixelLumi.PixelLumiDQM_cfi import pixel_lumi_dqm 

process.PixelLumiDqmAlcaLumiPixel = pixel_lumi_dqm

process.PixelLumiDqmZeroBias = pixel_lumi_dqm.clone(
    resetEveryNLumiSections=cms.untracked.int32(1),
    logFileName = cms.untracked.string("/nfshome0/dqmdev/pixel_lumi.txt")
    )
if process.dqmSaver.producer.value() is "Playback":
    process.PixelLumiDqmZeroBias.logFileName = cms.untracked.string("/nfshome0/dqmdev/pixel_lumi.txt")
else:
    process.PixelLumiDqmZeroBias.logFileName = cms.untracked.string("/nfshome0/dqmpro/pixel_lumi.txt")

    
process.PixelLumiDqmRandom = pixel_lumi_dqm.clone(
    resetEveryNLumiSections=cms.untracked.int32(20)
    )

process.PixelLumiDqmHFGetter = cms.EDAnalyzer("HFLumiGetter")

#----------------
# Lumi from "DIP"
#-------------------------------------------------
# process.DBService=cms.Service('DBService',
#         authPath= cms.untracked.string('/nfshome0/centraltspro/secure/')       
# )
process.GlobalTag.DBParameters.authenticationPath = cms.untracked.string('/nfshome0/centraltspro/secure/')
# process.DIPLumiProducer=cms.ESSource("DIPLumiProducer",
#                                      connect=cms.string('oracle://cms_omds_lb/CMS_RUNTIME_LOGGER')
#                                      #Zhen recommends:
# #        connect=cms.string('oracle://cms_orcon_prod/cms_lumi_prod')
# #        crashes at end of lumisection
#                                     )

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )
process.MessageLogger.cerr.threshold = cms.untracked.string("INFO")
process.MessageLogger.cerr.INFO.limit = cms.untracked.int32(-1)
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)

#-------------
# Input Source
#-------------
process.load("DQM.Integration.test.inputsource_cfi")
#process.DQMEventStreamHttpReader.consumerName = 'DQM Pixel Luminosity Consumer'
#process.DQMEventStreamHttpReader.SelectHLTOutput = cms.untracked.string('hltOutputALCALUMIPIXELS')
#process.DQMEventStreamHttpReader.maxEventRequestRate = cms.untracked.double(200.0)
#process.DQMEventStreamHttpReader.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('AlCa_LumiPixels_ZeroBias_v*'))


# Only unpack pixel data; there is nothing else, really.
process.MyRawToDigi = cms.Sequence(process.siPixelDigis)
# Change this to test without stable beams
process.siPixelDigis.InputLabel = cms.InputTag("hltFEDSelectorLumiPixels")
#process.siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")

# Only reconstruct the pixel clusters.
process.MyReconstruction = cms.Sequence(process.siPixelClusters)
process.reconstruction_step = cms.Path(process.MyReconstruction)

#-------------
# DQM services 
#-------------
process.dqm_step = cms.Path(process.dqmEnv*process.dqmSaver)

# Lumi DQM settings. These are from J's code, now default in the _cfi for PixelLumiDQM.
#process.pixel_lumi_tupler.includeVertexInfo = cms.untracked.bool(False)
#process.pixel_lumi_tupler.includeTrackInfo = cms.untracked.bool(False)
#process.pixel_lumi_tupler.includeStripClusterInfo = cms.untracked.bool(False)

#-------------------------------
# Path and schedule definitions.
#-------------------------------
process.raw2digi_step = cms.Path(process.MyRawToDigi)
process.HF = cms.Path(process.PixelLumiDqmHFGetter)
#process.plumdqm_alca_lumipixel_step = cms.Path(process.alca_lumi_pixels_zerobias_selector*
#                                               process.PixelLumiDqmAlcaLumiPixel)
process.plumdqm_alca_zerobias_step = cms.Path(
    #process.scraping_filter * # this seems to need tracks so cannot be used
    process.zerobias_selector *
    process.PixelLumiDqmZeroBias)
#
#process.plumdqm_alca_random_step = cms.Path(
#    process.scraping_filter * # this seems to need tracks so cannot be used
#                                            process.random_trigger_selector *
#                                            process.PixelLumiDqmRandom)

#--------------------
# Schedule definition
#--------------------
process.schedule = cms.Schedule(process.raw2digi_step,
                                process.reconstruction_step,
#                                process.HF,
                                process.plumdqm_alca_zerobias_step,
                                #process.plumdqm_alca_random_step,
                                process.dqm_step)


### process customizations included here
from DQM.Integration.test.online_customizations_cfi import *
process = customise(process)

######################################################################
