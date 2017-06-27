## 11.02.16: Copied from https://raw.githubusercontent.com/dcurry09/EMTF8/master/L1Trigger/L1TMuonEndCap/test/runMuonEndCap.py

import FWCore.ParameterSet.Config as cms
import os
import sys
import commands
import subprocess
from Configuration.StandardSequences.Eras import eras

process = cms.Process('L1TMuonEmulation')

## Import standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2016Reco_cff') ## Is this appropriate for 2015 data/MC? - AWB 18.04.16
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff') ## Will this work on 0T data? - AWB 18.04.16
process.load('Configuration.StandardSequences.RawToDigi_Data_cff') ## Will this work for MC? - AWB 18.04.16
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# ## Extra tools
# process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
#                                         ignoreTotal = cms.untracked.int32(1),
#                                         monitorPssAndPrivate = cms.untracked.bool(True))

## CSCTF digis, phi / pT LUTs?
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")

## Import RECO muon configurations
process.load("RecoMuon.TrackingTools.MuonServiceProxy_cff")
process.load("RecoMuon.TrackingTools.MuonTrackLoader_cff")

## Lumi JSON tools
import FWCore.PythonUtilities.LumiList as LumiList
# process.source.lumisToProcess = LumiList.LumiList(filename = 'goodList.json').getVLuminosityBlockRange()

## Message Logger and Event range
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000000) )  ## ~10k / 1 minute per file
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False))

process.options = cms.untracked.PSet(
    # SkipEvent = cms.untracked.vstring('ProductNotFound')
)

## Global Tags
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '') ## Good for 2015/2016 data/MC? - AWB 18.04.16
# process.GlobalTag = GlobalTag(process.GlobalTag, 'GR_P_V56', '') ## Used for anything? - AWB 18.04.16

## Event Setup Producer
process.load('L1Trigger.L1TMuonEndCap.fakeEmtfParams_cff') ## Why does this file have "fake" in the name? - AWB 18.04.16
                                                           ## for the data we should rely on the global tag mechanism - KK 2016.04.28
process.esProd = cms.EDAnalyzer("EventSetupRecordDataGetter",
                                toGet = cms.VPSet(
        ## Apparently L1TMuonEndcapParamsRcd doesn't exist in CondFormats/DataRecord/src/ (Important? - AWB 18.04.16)
        cms.PSet(record = cms.string('L1TMuonEndcapParamsRcd'),
                 data = cms.vstring('L1TMuonEndcapParams'))
        ),
                                verbose = cms.untracked.bool(True)
                                )


readFiles = cms.untracked.vstring()
secFiles  = cms.untracked.vstring()
process.source = cms.Source(
    'PoolSource',
    fileNames = readFiles,
    secondaryFileNames= secFiles
    #, eventsToProcess = cms.untracked.VEventRange('201196:265380261')
    )

eos_cmd = '/afs/cern.ch/project/eos/installation/pro/bin/eos.select'

## 2017 Cosmics, with RPC!
# in_dir_name = '/store/express/Commissioning2017/ExpressCosmics/FEVT/Express-v1/000/291/781/00000/'
# in_dir_name = '/store/express/Commissioning2017/ExpressCosmics/FEVT/Express-v1/000/291/891/00000/'
# in_dir_name = '/store/express/Commissioning2017/ExpressCosmics/FEVT/Express-v1/000/292/080/00000/'
# in_dir_name = '/store/express/Commissioning2017/ExpressCosmics/FEVT/Express-v1/000/292/497/00000/'
# in_dir_name = '/store/express/Commissioning2017/ExpressCosmics/FEVT/Express-v1/000/293/111/00000/'
in_dir_name = '/store/express/Commissioning2017/ExpressCosmics/FEVT/Express-v1/000/293/580/00000/'

# ## ZeroBias, IsolatedBunch data
# in_dir_name = '/store/data/Run2016H/ZeroBiasIsolatedBunch0/RAW/v1/000/282/650/00000/'

# ## SingleMu, Z-->mumu, high pT RECO muon
#in_dir_name = '/store/group/dpg_trigger/comm_trigger/L1Trigger/Data/Collisions/SingleMuon/Skims/200-pt-muon-skim_from-zmumu-skim-cmssw-8013/SingleMuon/'
#in_dir_name = in_dir_name+'crab_200-pt-muon-skim_from-zmumu-skim-cmssw-8013__SingleMuon_ZMu-2016B_v1/160710_225040/0000/'
#in_dir_name = in_dir_name+'crab_200-pt-muon-skim_from-zmumu-skim-cmssw-8013__SingleMuon_ZMu-2016B_v2/160710_225006/0000/'
#in_dir_name = in_dir_name+'crab_200-pt-muon-skim_from-zmumu-skim-cmssw-8013__SingleMuon_ZMu-2016C_v2/160710_225057/0000/'
#in_dir_name = in_dir_name+'crab_200-pt-muon-skim_from-zmumu-skim-cmssw-8013__SingleMuon_ZMu-2016C_v2/160710_225057/0001/'
#in_dir_name = in_dir_name+'crab_200-pt-muon-skim_from-zmumu-skim-cmssw-8013__SingleMuon_ZMu-2016D_v2/160710_225023/0000/'

nFiles = 0
for in_file_name in subprocess.check_output([eos_cmd, 'ls', in_dir_name]).splitlines():
    if not ('.root' in in_file_name): continue
    #if ( int(in_file_name.split('RECO_')[1].split('.roo')[0]) < 755 ): continue
    nFiles += 1
    ## if (nFiles % 10 != 0): continue ## Only process every 10th file
    readFiles.extend( cms.untracked.vstring(in_dir_name+in_file_name) )    

# readFiles.extend([
#         #'file:/afs/cern.ch/work/a/abrinke1/public/EMTF/Run2016G/RAW/279024/52622B4D-B265-E611-8099-FA163E326094.root'
#         ])

# secFiles.extend([
#         'root://eoscms.cern.ch//eoscms//eos/cms/store/data/Run2015B/SingleMuon/RAW/v1/000/251/168/00000/382EE8DB-2825-E511-B3E0-02163E013597.root'
#         ])


process.content = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")

process.load('EventFilter.L1TRawToDigi.emtfStage2Digis_cfi')

process.L1TMuonSeq = cms.Sequence(
    process.emtfStage2Digis
    )

process.L1TMuonPath = cms.Path(
    process.L1TMuonSeq
    )

# outCommands = cms.untracked.vstring('keep *')
outCommands = cms.untracked.vstring(

    'keep *_*emtf*_*_*',
    'keep *_*Emtf*_*_*',
    'keep *_*EMTF*_*_*',

    'keep recoMuons_muons__*',
    'keep CSCDetIdCSCCorrelatedLCTDigiMuonDigiCollection_*_*_*',
    'keep *_csctfDigis_*_*',
    'keep *_emtfStage2Digis_*_*',
    'keep *_simEmtfDigis_*_*',
    'keep *_gmtStage2Digis_*_*',
    'keep *_simGmtStage2Digis_*_*',

    )

out_dir = "/afs/cern.ch/work/a/abrinke1/public/EMTF/Commissioning/2017/"

process.out = cms.OutputModule("PoolOutputModule", 
                               # fileName = cms.untracked.string("EMTF_Tree_highPt200MuonSkim_2016D_debug.root"),
                               # fileName = cms.untracked.string("EMTF_Tree_ZeroBias_IsoBunch_282650_SingleHit_test.root"),
                               # fileName = cms.untracked.string(out_dir+"EMTF_Unpacked_Cosmics_291891_RPC_all_files.root"),
                               fileName = cms.untracked.string(out_dir+"EMTF_Unpacked_Cosmics_293580_RPC_100k.root"),
                               outputCommands = outCommands
                               )

process.output_step = cms.EndPath(process.out)

process.schedule = cms.Schedule(process.L1TMuonPath)

process.schedule.extend([process.output_step])

## What does this do? Necessary? - AWB 29.04.16
from SLHCUpgradeSimulations.Configuration.muonCustoms import customise_csc_PostLS1
process = customise_csc_PostLS1(process)
