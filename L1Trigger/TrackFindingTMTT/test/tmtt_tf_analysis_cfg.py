################################################################################################
# To run execute do
# cmsRun tmtt_tf_analysis_cfg.py Events=50 inputMC=Samples/Muons/PU0.txt histFile=outputHistFile.root 
# where the arguments take default values if you don't specify them. You can change defaults below.
#################################################################################################

import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
import FWCore.ParameterSet.VarParsing as VarParsing
import os

process = cms.Process("Demo")

GEOMETRY = "D88"

if GEOMETRY == "D88": 
    print("using geometry " + GEOMETRY + " (tilted)")
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
    process.load('Configuration.Geometry.GeometryExtended2026D88_cff')
else:
    print("this is not a valid geometry!!!")

process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.L1track = cms.untracked.PSet(limit = cms.untracked.int32(-1))

options = VarParsing.VarParsing ('analysis')

#--- Specify input MC

#--- To use MCsamples scripts, defining functions get*data*() for easy MC access,
#--- follow instructions in https://github.com/cms-L1TK/MCsamples

#from MCsamples.Scripts.getCMSdata_cfi import *
#from MCsamples.Scripts.getCMSlocaldata_cfi import *

if GEOMETRY == "D88":
  # Read data from card files (defines getCMSdataFromCards()):
  #from MCsamples.RelVal_1260_D88.PU200_TTbar_14TeV_cfi import *
  #inputMC = getCMSdataFromCards()

  # Or read .root files from directory on local computer:
  #dirName = "$myDir/whatever/"
  #inputMC=getCMSlocaldata(dirName)

  # Or read specified dataset (accesses CMS DB, so use this method only occasionally):
  #dataName="/RelValTTbar_14TeV/CMSSW_12_6_0-PU_125X_mcRun4_realistic_v5_2026D88PU200RV183v2-v1/GEN-SIM-DIGI-RAW"
  #inputMC=getCMSdata(dataName)

  # Or read specified .root file:
  inputMC = ["/store/mc/CMSSW_12_6_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_125X_mcRun4_realistic_v5_2026D88PU200RV183v2-v1/30000/0959f326-3f52-48d8-9fcf-65fc41de4e27.root"]

else:
  print("this is not a valid geometry!!!")


#--- Specify number of events to process.
options.register('Events',100,VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int,"Number of Events to analyze")

#--- Specify name of output histogram file. (If name = '', then no histogram file will be produced).
options.register('histFile','Hist.root',VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string,"Name of output histogram file")

#--- Specify if stubs need to be (re)made, e.g. because they are not available in the input file
options.register('makeStubs',0,VarParsing.VarParsing.multiplicity.singleton,VarParsing.VarParsing.varType.int,"Make stubs, and truth association, on the fly")

#--- Specify whether to output a GEN-SIM-DIGI-RAW dataset containing the TMTT L1 tracks & associators.
# (Warning: you may need to edit the associator python below to specify which track fitter you are using).
options.register('outputDataset',0,VarParsing.VarParsing.multiplicity.singleton,VarParsing.VarParsing.varType.int,"Create GEN-SIM-DIGI-RAW dataset containing TMTT L1 tracks")

options.parseArguments()

#--- input and output

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.Events) )

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(*inputMC))

outputHistFile = options.histFile

if outputHistFile != "":
  process.TFileService = cms.Service("TFileService", fileName = cms.string(outputHistFile))

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )
process.Timing = cms.Service("Timing", summaryOnly = cms.untracked.bool(True))

#--- Load code that produces our L1 tracks and makes corresponding histograms.
#process.load('L1Trigger.TrackFindingTMTT.TMTrackProducer_cff')

#--- Alternative cfg including improvements not yet in the firmware. Aimed at L1 trigger studies.
process.load('L1Trigger.TrackFindingTMTT.TMTrackProducer_Ultimate_cff')
#
# Enable histogramming & use of MC truth (considerably increases CPU)
process.TMTrackProducer.EnableMCtruth = True
process.TMTrackProducer.EnableHistos  = True
#
#--- Optionally override default configuration parameters here (example given of how).

#process.TMTrackProducer.TrackFitSettings.TrackFitters = [
#                                "KF5ParamsComb",
#                                "KF4ParamsComb"
#                                "KF4ParamsCombHLS",
#                                "ChiSquaredFit4",
#                                "SimpleLR4"
#                                ]

# If the input samples contain stubs and the truth association, then you can just use the following path
process.p = cms.Path(process.TMTrackProducer)

# Optionally (re)make the stubs on the fly
if options.makeStubs == 1:
        process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')
        process.load('SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff')
        process.TTClusterAssociatorFromPixelDigis.digiSimLinks = cms.InputTag("simSiPixelDigis","Tracker")

        LOOSE_STUBS = False

        if (LOOSE_STUBS):
          # S.Viret's loose Dec. 2017 stub windows from commented out part of
          # L1Trigger/TrackTrigger/python/TTStubAlgorithmRegister_cfi.py in CMSSW 9.3.7
          # Optimised for electrons.

          process.TTStubAlgorithm_official_Phase2TrackerDigi_ = cms.ESProducer("TTStubAlgorithm_official_Phase2TrackerDigi_",
          zMatchingPS  = cms.bool(True),
          zMatching2S  = cms.bool(True),
          #Number of tilted rings per side in barrel layers (for tilted geom only)
          NTiltedRings = cms.vdouble( 0., 12., 12., 12., 0., 0., 0.), 
          BarrelCut    = cms.vdouble( 0, 2.0, 3, 4.5, 6, 6.5, 7.0),
          TiltedBarrelCutSet = cms.VPSet(
               cms.PSet( TiltedCut = cms.vdouble( 0 ) ),
               cms.PSet( TiltedCut = cms.vdouble( 0, 3, 3., 2.5, 3., 3., 2.5, 2.5, 2., 1.5, 1.5, 1, 1) ),
               cms.PSet( TiltedCut = cms.vdouble( 0, 4., 4, 4, 4, 4., 4., 4.5, 5, 4., 3.5, 3.5, 3) ),
               cms.PSet( TiltedCut = cms.vdouble( 0, 5, 5, 5, 5, 5, 5, 5.5, 5, 5, 5.5, 5.5, 5.5) ),
              ),
          EndcapCutSet = cms.VPSet(
               cms.PSet( EndcapCut = cms.vdouble( 0 ) ),
               cms.PSet( EndcapCut = cms.vdouble( 0, 1., 2.5, 2.5, 3.5, 5.5, 5.5, 6, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 7, 7) ),
               cms.PSet( EndcapCut = cms.vdouble( 0, 0.5, 2.5, 2.5, 3, 5, 6, 6, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 7, 7) ),
               cms.PSet( EndcapCut = cms.vdouble( 0, 1, 3., 4.5, 6., 6.5, 6.5, 6.5, 7, 7, 7, 7, 7) ),
               cms.PSet( EndcapCut = cms.vdouble( 0, 1., 2.5, 3.5, 6., 6.5, 6.5, 6.5, 6.5, 7, 7, 7, 7) ),
               cms.PSet( EndcapCut = cms.vdouble( 0, 0.5, 1.5, 3., 4.5, 6.5, 6.5, 7, 7, 7, 7, 7, 7) ),
              )
          )

        else:
          # S.Viret's July 2017 stub windows (tight) from commented out part of
          # L1Trigger/TrackTrigger/python/TTStubAlgorithmRegister_cfi.py in CMSSW 9.3.2

          process.TTStubAlgorithm_official_Phase2TrackerDigi_ = cms.ESProducer("TTStubAlgorithm_official_Phase2TrackerDigi_",
          zMatchingPS  = cms.bool(True),
          zMatching2S  = cms.bool(True),
          #Number of tilted rings per side in barrel layers (for tilted geom only)
          NTiltedRings = cms.vdouble( 0., 12., 12., 12., 0., 0., 0.), 
          BarrelCut = cms.vdouble( 0, 2.0, 2.0, 3.5, 4.5, 5.5, 6.5), #Use 0 as dummy to have direct access using DetId to the correct element 

          TiltedBarrelCutSet = cms.VPSet(
              cms.PSet( TiltedCut = cms.vdouble( 0 ) ),
              cms.PSet( TiltedCut = cms.vdouble( 0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2., 2., 1.5, 1.5, 1., 1.) ),
              cms.PSet( TiltedCut = cms.vdouble( 0, 3., 3., 3., 3., 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2, 2) ),
              cms.PSet( TiltedCut = cms.vdouble( 0, 4.5, 4.5, 4, 4, 4, 4, 3.5, 3.5, 3.5, 3, 3, 3) ),
              ),
          EndcapCutSet = cms.VPSet(
              cms.PSet( EndcapCut = cms.vdouble( 0 ) ),
              cms.PSet( EndcapCut = cms.vdouble( 0, 1, 1.5, 1.5, 2, 2, 2.5, 3, 3, 3.5, 4, 2.5, 3, 3.5, 4.5, 5.5) ),
              cms.PSet( EndcapCut = cms.vdouble( 0, 1, 1.5, 1.5, 2, 2, 2, 2.5, 3, 3, 3, 2, 3, 4, 5, 5.5) ),
              cms.PSet( EndcapCut = cms.vdouble( 0, 1.5, 1.5, 2, 2, 2.5, 2.5, 2.5, 3.5, 2.5, 5, 5.5, 6) ),
              cms.PSet( EndcapCut = cms.vdouble( 0, 1.0, 1.5, 1.5, 2, 2, 2, 2, 3, 3, 6, 6, 6.5) ),
              cms.PSet( EndcapCut = cms.vdouble( 0, 1.0, 1.5, 1.5, 1.5, 2, 2, 2, 3, 3, 6, 6, 6.5) ),
              )
          ) 

        process.p = cms.Path(process.TrackTriggerClustersStubs * process.TrackTriggerAssociatorClustersStubs * process.TMTrackProducer)

# Optionally create output GEN-SIM-DIGI-RAW dataset containing TMTT L1 tracks & associators.
if options.outputDataset == 1:

  # Associate TMTT L1 tracks to truth particles
  process.load('SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff')
  process.TTAssociatorTMTT = process.TTTrackAssociatorFromPixelDigis.clone(
# Edit to specify which input L1 track collection to run associator on.
          TTTracks = cms.VInputTag(cms.InputTag("TMTrackProducer", 'TML1TracksKF4ParamsComb'))
#          TTTracks = cms.VInputTag(cms.InputTag("TMTrackProducer", 'TML1TracksglobalLinearRegression'))
  )
  process.pa = cms.Path(process.TTAssociatorTMTT)

  # Write output dataset
  process.load('Configuration.EventContent.EventContent_cff')

  process.writeDataset = cms.OutputModule("PoolOutputModule",
      splitLevel = cms.untracked.int32(0),
      eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
      outputCommands = process.RAWSIMEventContent.outputCommands,
      fileName = cms.untracked.string('output_dataset.root'), ## ADAPT IT ##
      dataset = cms.untracked.PSet(
          filterName = cms.untracked.string(''),
          dataTier = cms.untracked.string('GEN-SIM')
      )
  )
  # Include TMTT L1 tracks & associators + stubs.
  process.writeDataset.outputCommands.append('keep  *TTTrack*_*_*_*')
  process.writeDataset.outputCommands.append('keep  *TTStub*_*_*_*')

  process.pd = cms.EndPath(process.writeDataset)
  process.schedule = cms.Schedule(process.p, process.pa, process.pd)
