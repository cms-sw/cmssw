from __future__ import print_function
import os
import FWCore.ParameterSet.Config as cms


##
## Setup command line options
##
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
options = VarParsing.VarParsing ('standard')
options.register('sample', 'data1', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Input sample")

# get and parse the command line arguments
options.parseArguments()

print("Input sample: ", options.sample)


##
## Process definition
##
process = cms.Process("ApeSkim")



##
## Message Logger
##
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.AlignmentTrackSelector=dict()
process.MessageLogger.cerr.INFO.limit = 0
process.MessageLogger.cerr.default.limit = -1
process.MessageLogger.cerr.AlignmentTrackSelector = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.FwkReport.reportEvery = 1000 ## really show only every 1000th



##
## Process options
##
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
)


##
## Start of Configuration
##

maxEvents = -1
outputName = "defaultOutputName.root"
outputPath = None
outputFileSize = 350000

##
## TrackSelection can be SingleMu, DoubleMu, MinBias, Cosmics
## The choice affects which AlignmentTrackSelector is used.
## Currently, DoubleMu means ZToMuMu, so if there is the need 
## for UpsilonToMuMu or JPsiToMuMu, these have to be added first
##
trackSelection = "SingleMu"
globalTag = None
outputPath = None # can also be specified. If that is done, files are copied to this path afterwards

if "iov" in options.sample:
    ## Configure here for campaigns with many different datasets (such as multi-IOV)
    iovNo = options.sample.split("iov")[1]
    process.load("Alignment.APEEstimation.samples.")
    outputName = ".root"
    outputPath = None
    trackSelection = "SingleMu"
if options.sample == 'data1':
    process.load("Alignment.APEEstimation.samples.Data_TkAlMinBias_Run2018C_PromptReco_v3_cff")
    outputName = 'MinBias.root'
    #outputPath = "workingArea"
    trackSelection = "MinBias"
if options.sample == 'data3':
    process.load("Alignment.APEEstimation.samples.Data_TkAlMuonIsolated_22Jan2013C_v1_cff")
    outputName = 'Data_TkAlMuonIsolated_22Jan2013C.root'
    trackSelection = "SingleMu"
if options.sample == 'data4':
    process.load("Alignment.APEEstimation.samples.Data_TkAlMuonIsolated_22Jan2013D_v1_cff")
    outputName = 'Data_TkAlMuonIsolated_22Jan2013D.root'
    trackSelection = "SingleMu"
# The following options are used for MC samples
if options.sample == 'qcd':
    globalTag = "auto:run2_mc"
    process.load("Alignment.APEEstimation.samples.MC_UL16_ttbar_cff")
    outputPath = '/eos/cms/store/caf/user/mteroerd/Skims/MC/UL16'    
    outputName = 'MC_UL16_ttbar.root'
    trackSelection = "GenSim"
if options.sample == 'wlnu':
    process.load("Alignment.APEEstimation.samples.Mc_TkAlMuonIsolated_2016UL_cff")
    outputPath = '/eos/cms/store/caf/user/jschulz/Skims/MC/UL2016ReRecoRealistic'
    outputName = 'Mc_TkAlMuonIsolated_WJetsToLNu_2016.root'
    trackSelection = "SingleMu"
    
# For unit tests
if options.sample == 'UnitTest':
    process.load("Alignment.APEEstimation.samples.MC_UnitTest_TkAlMuonIsolated_cff")
    outputName = 'MC_UnitTest_TkAlMuonIsolated.root'
    maxEvents = 1000
    globalTag = "auto:phase1_2022_design"
    trackSelection = "SingleMu"


print("Using output name %s"%(outputName))
if outputPath:
    print("Using output path %s"%(outputPath))

##
## Choice of GlobalTag
##

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag

if globalTag == None:
    print("No global tag specified, is this intended?")
else:   
    process.GlobalTag = GlobalTag(process.GlobalTag, globalTag, '')
print("Using global tag "+process.GlobalTag.globaltag._value)

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

##
## Number of Events (should be after input file)
##
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(maxEvents) )


##
## Skim tracks
##



import Alignment.APEEstimation.AlignmentTrackSelector_cff as AlignmentTrackSelector


# Determination of which AlignmentTrackSelector to use
if trackSelection == "SingleMu":
    trackSelector = AlignmentTrackSelector.MuSkimSelector
elif trackSelection == "GenSim":
    trackSelector = AlignmentTrackSelector.genSimSkimSelector    
elif trackSelection == "DoubleMu":
    trackSelector = AlignmentTrackSelector.DoubleMuSkimSelector
elif trackSelection == "MinBias":
    trackSelector = AlignmentTrackSelector.MinBiasSkimSelector
elif trackSelection == "Cosmics":
    trackSelector = AlignmentTrackSelector.CosmicsSkimSelector
else: # Extend list here with custom track selectors
    print("Unknown trackSelection %s, exiting"%(trackSelection))
    exit(1)

process.MuSkim = trackSelector

import Alignment.CommonAlignment.tools.trackselectionRefitting as trackselRefit
process.seqTrackselRefit = trackselRefit.getSequence(process, trackSelector.src.getModuleLabel())


##
## Path
##
process.path = cms.Path(
    process.offlineBeamSpot*
    process.seqTrackselRefit*
    process.MuSkim
)

##
## Define event selection from path
##
EventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('path')
    )
)


##
## configure output module
##
process.out = cms.OutputModule("PoolOutputModule",
    ## Parameters directly for PoolOutputModule
    fileName = cms.untracked.string(outputName),
    # Maximum size per file before a new one is created
    maxSize = cms.untracked.int32(outputFileSize),
    dropMetaData = cms.untracked.string("DROPPED"),
    ## Parameters for inherited OutputModule
    SelectEvents = EventSelection.SelectEvents,
    outputCommands = cms.untracked.vstring(
        'drop *',
    ),
)
process.load("Alignment.APEEstimation.PrivateSkim_EventContent_cff")
process.out.outputCommands.extend(process.ApeSkimEventContent.outputCommands)


##
## Outpath
##
process.outpath = cms.EndPath(process.out)
