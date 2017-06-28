
# ## Basic (but usually unnecessary) imports
# import os
# import sys
# import commands

import subprocess

import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('reL1T', eras.Run2_2016)

## Import standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi') ## What does this module do? - AWB 05.09.16
process.load('Configuration.StandardSequences.GeometryRecoDB_cff') ## What does this module do? - AWB 05.09.16
process.load('Configuration.StandardSequences.MagneticField_cff') ## Different than in data ("MagneticField_AutoFromDBCurrent_cff"?)
process.load('Configuration.StandardSequences.SimL1EmulatorRepack_FullMC_cff') ## Different than in data
process.load('Configuration.StandardSequences.RawToDigi_cff') ## Different than in data ("RawToDigi_Data_cff"?)
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi')

## CSCTF digis, phi / pT LUTs?
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")

## Import RECO muon configurations
process.load("RecoMuon.TrackingTools.MuonServiceProxy_cff")
process.load("RecoMuon.TrackingTools.MuonTrackLoader_cff")

## Message Logger and Event range
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False))

## Global Tags
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

## Default parameters for firmware version, pT LUT XMLs, and coordinate conversion LUTs
process.load('L1Trigger.L1TMuonEndCap.fakeEmtfParams_cff')


readFiles = cms.untracked.vstring()
process.source = cms.Source(
    'PoolSource',
    fileNames = readFiles
    )

# eos_cmd = '/afs/cern.ch/project/eos/installation/pro/bin/eos.select'

# in_dir_name = '/store/relval/CMSSW_8_0_19/RelValNuGun_UP15/GEN-SIM-DIGI-RECO/PU25ns_80X_mcRun2_asymptotic_2016_TrancheIV_v2_Tr4GT_v2_FastSim-v1/00000/'

# for in_file_name in subprocess.check_output([eos_cmd, 'ls', in_dir_name]).splitlines():
#     if not ('.root' in in_file_name): continue
#     readFiles.extend( cms.untracked.vstring(in_dir_name+in_file_name) )

dir_str = 'root://cms-xrd-global.cern.ch/'
dir_str += '/store/mc/RunIISpring16DR80/SingleMu_Pt1To1000_FlatRandomOneOverPt/GEN-SIM-RAW/NoPURAW_NZS_withHLT_80X_mcRun2_asymptotic_v14-v1/60000/'

readFiles.extend([
        
        # ## Tau-to-3-mu MC
        # 'root://eoscms.cern.ch//eos/cms/store/user/wangjian/DsTau3Mu_FullSim_1007/merged_fltr.root'

        ## SingleMu MC, noPU, flat in 1/pT
        ## DAS: dataset=/SingleMu_Pt1To1000_FlatRandomOneOverPt/RunIISpring16DR80-NoPURAW_NZS_withHLT_80X_mcRun2_asymptotic_v14-v1/GEN-SIM-RAW

        'file:/afs/cern.ch/work/a/abrinke1/public/EMTF/MC/SingleMu_Pt1To1000_FlatRandomOneOverPt/26CA310A-4164-E611-BE48-001E67248566.root',

        # dir_str+'26CA310A-4164-E611-BE48-001E67248566.root',
        # dir_str+'2C138BAC-4164-E611-8D9C-001E6724865B.root',
        # dir_str+'4CF53555-4164-E611-8347-001E67248566.root',
        # dir_str+'7E409BA3-3D64-E611-B392-001E67248142.root',
        # dir_str+'AE863F4E-3764-E611-977A-001E67248A25.root',
        # dir_str+'E65524B4-E964-E611-805F-0CC47A7139C4.root',

        # dir_str+'14C3C3CB-3864-E611-B770-20CF3027A624.root',
        # dir_str+'4645B2E7-3864-E611-9724-5404A64A1265.root',
        # dir_str+'507EEB74-3964-E611-B03B-3417EBE5361A.root',
        # dir_str+'683F88CB-4464-E611-AE40-20CF3027A5D8.root',
        # dir_str+'74715FCC-3864-E611-AA0F-3417EBE539DA.root',
        # dir_str+'769D60D3-3964-E611-8D1C-44A8423D7989.root',
        # dir_str+'E4541CA0-3664-E611-9DC1-00221982B650.root',

        # dir_str+'401C4E92-3F64-E611-B190-00259090765E.root',
        # dir_str+'644DC20B-3464-E611-9858-0025909083EE.root',
        # dir_str+'A6DBC3A6-E964-E611-8798-002590907826.root',
        # dir_str+'D24C4348-3664-E611-BFB7-00259090784E.root',
        # dir_str+'E8BE669B-3C64-E611-9170-00259090766E.root',
        # dir_str+'EEF06A5C-3564-E611-8F72-002590908EC2.root',
        # dir_str+'F04B42E3-3D64-E611-AA23-0025909083EE.root',
        # dir_str+'FCC1A3C5-4064-E611-8812-002590907836.root',

        ])

process.content = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")


# Path and EndPath definitions

# ## Defined in Configuration/StandardSequences/python/SimL1EmulatorRepack_FullMC_cff.py
# process.L1RePack_step = cms.Path(process.SimL1Emulator)
SimL1Emulator_AWB = cms.Sequence(process.unpackRPC+process.unpackCSC)
process.L1RePack_step = cms.Path(SimL1Emulator_AWB)

## Defined in Configuration/StandardSequences/python/RawToDigi_cff.py
## Includes L1TRawToDigi, defined in L1Trigger/Configuration/python/L1TRawToDigi_cff.py
# process.raw2digi_step = cms.Path(process.RawToDigi)

process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag('unpackCSC', 'MuonCSCComparatorDigi')
process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag('unpackCSC', 'MuonCSCWireDigi')

process.load('L1Trigger.L1TMuonEndCap.simEmtfDigis_cfi')

process.simEmtfDigis.verbosity = cms.untracked.int32(0)

##############################################
###  Settings for 2016 vs. 2017 emulation  ###
##############################################

# ## *** 2016 ***
# ## From python/fakeEmtfParams_cff.py
# process.emtfParams.PtAssignVersion = cms.int32(5)
# process.emtfParams.FirmwareVersion = cms.int32(49999) ## Settings as of end-of-year 2016
# process.emtfParams.PrimConvVersion = cms.int32(0)
# process.emtfForestsDB.toGet = cms.VPSet(
#     cms.PSet(
#         record = cms.string("L1TMuonEndCapForestRcd"),
#         tag = cms.string("L1TMuonEndCapForest_static_2016_mc")
#         )
#     )

# ## From python/simEmtfDigis_cfi.py
# process.simEmtfDigis.RPCEnable                 = cms.bool(False)
# process.simEmtfDigis.spTBParams16.ThetaWindow  = cms.int32(4)
# process.simEmtfDigis.spPCParams16.FixME11Edges = cms.bool(False)
# process.simEmtfDigis.spPAParams16.PtLUTVersion = cms.int32(5)
# process.simEmtfDigis.spPAParams16.BugGMTPhi    = cms.bool(True)


# ## *** 2017 ***
# ## From python/fakeEmtfParams_cff.py
# process.emtfParams.PtAssignVersion = cms.int32(7)
# process.emtfParams.FirmwareVersion = cms.int32(50001) ## Settings as of beginning-of-year 2017
# process.emtfParams.PrimConvVersion = cms.int32(1)
# # process.emtfForestsDB.toGet = cms.VPSet(
# #     cms.PSet(
# #         record = cms.string("L1TMuonEndCapForestRcd"),
# #         tag = cms.string("L1TMuonEndCapForest_static_Sq_20170523_mc")
# #         )
# #     )

# ## From python/simEmtfDigis_cfi.py
# process.simEmtfDigis.RPCEnable                 = cms.bool(True)
# process.simEmtfDigis.spTBParams16.ThetaWindow  = cms.int32(8)
# process.simEmtfDigis.spPCParams16.FixME11Edges = cms.bool(True)
# process.simEmtfDigis.spPAParams16.PtLUTVersion = cms.int32(7)
# process.simEmtfDigis.spPAParams16.BugGMTPhi    = cms.bool(False)

# process.simEmtfDigis.spPAParams16.ReadPtLUTFile = cms.bool(True)


# RawToDigi_AWB = cms.Sequence(process.muonCSCDigis+process.muonRPCDigis+process.csctfDigis)
RawToDigi_AWB = cms.Sequence(process.simCscTriggerPrimitiveDigis+process.muonCSCDigis+process.muonRPCDigis+process.csctfDigis+process.simEmtfDigis)
# RawToDigi_AWB = cms.Sequence(process.simMuonCSCDigis+process.cscTriggerPrimitiveDigis+process.muonCSCDigis+process.muonRPCDigis+process.cscTriggerPrimitiveDigis+process.csctfDigis+process.simEmtfDigis)
process.raw2digi_step = cms.Path(RawToDigi_AWB)

## Defined in Configuration/StandardSequences/python/EndOfProcess_cff.py
process.endjob_step = cms.EndPath(process.endOfProcess)

# process.L1TMuonSeq = cms.Sequence(
#     process.muonCSCDigis + ## Unpacked CSC LCTs from TMB
#     process.csctfDigis + ## Necessary for legacy studies, or if you use csctfDigis as input
#     process.muonRPCDigis +
#     ## process.esProd + ## What do we loose by not having this? - AWB 18.04.16
#     process.emtfStage2Digis +
#     process.simEmtfDigis
#     ## process.ntuple
#     )

# process.L1TMuonPath = cms.Path(
#     process.L1TMuonSeq
#     )

# outCommands = cms.untracked.vstring('keep *')
outCommands = cms.untracked.vstring(

    'keep recoMuons_muons__*',
    'keep *Gen*_*_*_*',
    'keep *_*Gen*_*_*',
    'keep *gen*_*_*_*',
    'keep *_*gen*_*_*',
    'keep CSCDetIdCSCCorrelatedLCTDigiMuonDigiCollection_*_*_*', ## muonCSCDigis
    'keep RPCDetIdRPCDigiMuonDigiCollection_*_*_*', ## muonRPCDigis
    #'keep CSCCorrelatedLCTDigiCollection_muonCSCDigis_*_*',
    #'keep *_*_*muonCSCDigis*_*',
    #'keep *_*_*_*muonCSCDigis*',
    'keep *_csctfDigis_*_*',
    'keep *_emtfStage2Digis_*_*',
    'keep *_simEmtfDigis_*_*',
    'keep *_simEmtfDigis_*_*',
    'keep *_simEmtfDigisMC_*_*',
    'keep *_gmtStage2Digis_*_*',
    'keep *_simGmtStage2Digis_*_*',

    )

out_dir = "/afs/cern.ch/work/a/abrinke1/public/EMTF/Commissioning/2017/"
# out_dir = "./"

process.treeOut = cms.OutputModule("PoolOutputModule", 
                                   # fileName = cms.untracked.string("EMTF_MC_Tree_RelValNuGun_UP15_1k.root"),
                                   # fileName = cms.untracked.string("EMTF_MC_Tree_tau_to_3_mu_RPC_debug.root"),
                                   # fileName = cms.untracked.string(out_dir+"EMTF_MC_Tree_SingleMu_2017_fromXMLv7_test.root"),
                                   fileName = cms.untracked.string(out_dir+"EMTF_MC_Tree_SingleMu_2017_v7_O2O_v5_11k.root"),
                                   outputCommands = outCommands
                                   )

process.treeOut_step = cms.EndPath(process.treeOut) ## Keep output tree - AWB 08.07.16

# Schedule definition
process.schedule = cms.Schedule(process.L1RePack_step,process.raw2digi_step,process.endjob_step,process.treeOut_step)

# process.output_step = cms.EndPath(process.treeOut)
# process.schedule = cms.Schedule(process.L1TMuonPath)
# process.schedule.extend([process.output_step])

# ## What does this do? Necessary? - AWB 29.04.16
# from SLHCUpgradeSimulations.Configuration.muonCustoms import customise_csc_PostLS1
# process = customise_csc_PostLS1(process)
