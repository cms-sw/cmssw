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
options.register('useTrackList', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Use list of preselected tracks")
options.register('isTest', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Test run")

# get and parse the command line arguments
options.parseArguments()

print("Input sample: ", options.sample)
print("Use list of preselected tracks: ", options.useTrackList)
print("Test run: ", options.isTest)


##
## Process definition
##
process = cms.Process("ApeSkim")



##
## Message Logger
##
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.categories.append('AlignmentTrackSelector')
#process.MessageLogger.categories.append('')
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



isData1 = isData2 = isData3 = isData4 = False
isMultiIOV = False
isData = False
isQcd = isWlnu = isZmumu = isZtautau = isZmumu10 = isZmumu20 =  isZmumu50 = False
isMc = False
if "iov" in  options.sample:
    isMultiIOV = True
    isData = True
if options.sample == 'data1':
    isData1 = True
    isData = True
elif options.sample == 'data2':
    isData2 = True
    isData = True
elif options.sample == 'data3':
    isData3 = True
    isData = True
elif options.sample == 'data4':
    isData4 = True
    isData = True
elif options.sample == 'qcd':
    isQcd = True
    isMc = True
elif options.sample == 'wlnu':
    isWlnu = True
    isMc = True
elif options.sample == 'zmumu':
    isZmumu = True
    isMc = True
elif options.sample == 'ztautau':
    isZtautau = True
    isMc = True
elif options.sample == 'zmumu10':
    isZmumu10 = True
    isMc = True
elif options.sample == 'zmumu20':
    isZmumu20 = True
    isMc = True
elif options.sample == 'zmumu50':
    isZmumu50 = True
    isMc = True

else:
    print('ERROR --- incorrect data sammple: ', options.sample)
    exit(8888)



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

if isMultiIOV:
    ## Configure here for campaigns with many different datasets (such as multi-IOV)
    iovNo = int(options.sample.split("iov")[1])
    process.load("Alignment.APEEstimation.samples.")
    outputName = ".root"
    outputPath = None # can also be specified. If that is done, files are copied to this path afterwards
    trackSelection = "SingleMu"
if isData1: 
    process.load("Alignment.APEEstimation.samples.Data_TkAlMinBias_Run2018C_PromptReco_v3_cff")
    outputName = 'MinBias.root'
    #outputPath = "workingArea"
    trackSelection = "MinBias"
if isData2: 
    process.load("Alignment.APEEstimation.samples.Data_TkAlMinBias_Run2018C_PromptReco_v3_cff")
    outputName = 'MinBias1.root'
    #outputPath = "workingArea"
    trackSelection = "MinBias"
if isData3: 
    process.load("Alignment.APEEstimation.samples.Data_TkAlMuonIsolated_22Jan2013C_v1_cff")
    outputName = 'Data_TkAlMuonIsolated_22Jan2013C.root'
    trackSelection = "SingleMu"
if isData4: 
    process.load("Alignment.APEEstimation.samples.Data_TkAlMuonIsolated_22Jan2013D_v1_cff")
    outputName = 'Data_TkAlMuonIsolated_22Jan2013D.root'
    trackSelection = "SingleMu"
# The following options are used for MC samples
if isQcd: 
    process.load("Alignment.APEEstimation.samples.Mc_TkAlMuonIsolated_Summer12_qcd_cff")
    outputName = 'Mc_TkAlMuonIsolated_Summer12_qcd.root'
    trackSelection = "MinBias"
if isWlnu: 
    process.load("Alignment.APEEstimation.samples.Mc_WJetsToLNu_74XTest_cff")
    outputName = 'Mc_WJetsToLNu_74XTest.root'
    trackSelection = "SingleMu"
if isZmumu: 
    process.load("")
    outputName = ''
    trackSelection = "DoubleMu"
if isZmumu10: 
    process.load("Alignment.APEEstimation.samples.Mc_TkAlMuonIsolated_Summer12_zmumu10_cff")
    outputName = 'Mc_TkAlMuonIsolated_Summer12_zmumu10.root'
    trackSelection = "DoubleMu"
if isZmumu20: 
    process.load("Alignment.APEEstimation.samples.Mc_TkAlMuonIsolated_Summer12_zmumu20_cff")
    outputName = 'Mc_TkAlMuonIsolated_Summer12_zmumu20.root'
    trackSelection = "DoubleMu"
if isZmumu50: 
    process.load("Alignment.APEEstimation.samples.DYToMuMu_M-50_Tune4C_13TeV-pythia8_Spring14dr-TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1_ALCARECO_cff")
    outputName = 'Mc_DYToMuMu_M-50_Tune4C_13TeV-pythia8_Spring14dr-TkAlMuonIsolated-castor_PU_S14_POSTLS170_V6-v1.root'
    trackSelection = "DoubleMu"


print("Using output name %s"%(outputName))
if outputPath:
    print("Using output path %s"%(outputPath))

##
## Choice of GlobalTag
##

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag

if isData:
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
    #process.GlobalTag = GlobalTag(process.GlobalTag, '101X_dataRun2_Prompt_v11', '')
elif isMc:
    #process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_design', '')

print("Using global tag "+process.GlobalTag.globaltag._value)

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

##
## Number of Events (should be after input file)
##
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(maxEvents) )
if options.isTest: process.maxEvents.input = 1001


##
## Skim tracks
##



import Alignment.APEEstimation.AlignmentTrackSelector_cff as AlignmentTrackSelector


# Determination of which AlignmentTrackSelector to use
if trackSelection == "SingleMu":
    trackSelector = AlignmentTrackSelector.MuSkimSelector
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

##
## If preselected track list is used
##
if options.useTrackList:
    process.MuSkim.src = 'TrackList'
    process.TriggerSelectionSequence *= process.TrackList

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


if options.isTest:
  process.out.fileName = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/test_apeSkim.root'


##
## Outpath
##
process.outpath = cms.EndPath(process.out)
