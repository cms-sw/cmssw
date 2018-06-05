import FWCore.ParameterSet.Config as cms
import os

##############################################################################
# customisations for L1 ntuple generation
#
# Add new customisations to this file!
#
# Example usage :
#   cmsDriver.py testNtuple -s NONE --customise=L1Trigger/L1TNtuples/customiseL1Ntuple.customiseL1NtupleAOD --conditions=auto:run2_mc_50ns --filein='/store/relval/CMSSW_7_5_0_pre1/RelValProdTTbar_13/AODSIM/MCRUN2_74_V7-v1/00000/48159643-5EE3-E411-818F-0025905A48F0.root' -n 100
#
##############################################################################

def L1NtupleTFileOut(process):

    process.TFileService = cms.Service(
        "TFileService",
        fileName = cms.string('L1Ntuple.root')
    )

    return process

from L1Trigger.L1TNtuples.customiseL1CustomReco import *
        

def L1NtupleAOD(process):
    
    L1NtupleTFileOut(process)
    L1NtupleCustomReco(process)

    process.load('L1Trigger.L1TNtuples.L1NtupleAOD_cff')
    process.l1ntupleaod = cms.Path(
        process.L1NtupleAOD
    )

    process.schedule.append(process.l1ntupleaod)

    return process

def L1NtupleAODCalo(process):

    L1NtupleTFileOut(process)
    L1NtupleCustomReco(process)

    process.load('L1Trigger.L1TNtuples.L1NtupleAODCalo_cff')
    process.l1ntupleaodcalo = cms.Path(
        process.L1NtupleAODCalo
    )

    process.schedule.append(process.l1ntupleaodcalo)

    return process


def L1NtupleAOD_MC(process):
    
    L1NtupleAOD(process)

    process.l1JetRecoTree.jecToken = cms.untracked.InputTag("ak4PFCHSL1FastL2L3Corrector")

    return process



def L1NtupleRAW(process):

    L1NtupleTFileOut(process)

    process.load('L1Trigger.L1TNtuples.L1NtupleRAW_cff')
    process.l1ntupleraw = cms.Path(
        process.L1NtupleRAW
    )

    process.schedule.append(process.l1ntupleraw)

    # for 5 BX of candidates in L1Extra
    if "gctDigis" in process.producers:
        process.gctDigis.numberOfGctSamplesToUnpack = cms.uint32(5)

    if "l1extraParticles" in process.producers:
        process.l1extraParticles.centralBxOnly = cms.bool(False)

    return process

def L1NtupleNANO(process):

    L1NtupleTFileOut(process)

    process.load('L1Trigger.L1TNtuples.L1NtupleNANO_cff')
    process.l1ntuplenano = cms.Path(
        process.L1NtupleNANO
    )

    process.schedule.append(process.l1ntuplenano)

    return process

def L1NtupleRAWCalo(process):

    L1NtupleTFileOut(process)

    process.load('L1Trigger.L1TNtuples.L1NtupleRAWCalo_cff')
    process.l1ntuplerawcalo = cms.Path(
        process.L1NtupleRAWCalo
    )

    process.schedule.append(process.l1ntuplerawcalo)

    return process


def L1NtupleEMU(process):

    L1NtupleTFileOut(process)

    process.load('L1Trigger.L1TNtuples.L1NtupleEMU_cff')
    process.l1ntupleemu = cms.Path(
        process.L1NtupleEMU
    )

    process.schedule.append(process.l1ntupleemu)

    return process

def L1NtupleEMUCalo(process):

    L1NtupleTFileOut(process)

    process.load('L1Trigger.L1TNtuples.L1NtupleEMUCalo_cff')
    process.l1ntupleemucalo = cms.Path(
        process.L1NtupleEMUCalo
    )

    process.schedule.append(process.l1ntupleemucalo)

    return process


def L1NtupleEMULegacy(process):

    L1NtupleTFileOut(process)

    process.load('L1Trigger.L1TNtuples.L1NtupleEMULegacy_cff')
    process.l1ntupleemulegacy = cms.Path(
        process.L1NtupleEMULegacy
    )

    process.schedule.append(process.l1ntupleemulegacy)

    return process


def L1NtupleGEN(process):

    L1NtupleTFileOut(process)

    process.load('L1Trigger.L1TNtuples.L1NtupleGEN_cff')
    process.l1ntuplegen = cms.Path(
        process.L1NtupleGEN
    )

    process.schedule.append(process.l1ntuplegen)

    return process


def L1NtupleRAWEMU(process):

    L1NtupleRAW(process)
    L1NtupleEMU(process)

    return process

def L1NtupleRAWEMUCalo(process):

    L1NtupleRAWCalo(process)
    L1NtupleEMUCalo(process)

    return process

def L1NtupleNANOEMU(process):

    L1NtupleNANO(process)
    process.load('L1Trigger.L1TNtuples.L1NtupleEMU_cff')

    process.L1NtupleGTEMU = cms.Sequence( process.l1uGTEmuTree )
    process.l1ntuplegtemu = cms.Path(
        process.L1NtupleGTEMU
    )
    process.schedule.append(process.l1ntuplegtemu)

    return process

def L1NtupleRAWEMULegacy(process):

    L1NtupleRAW(process)
    L1NtupleEMU(process)
    L1NtupleEMULegacy(process)

    return process


def L1NtupleAODRAW(process):

    L1NtupleRAW(process)
    L1NtupleAOD(process)

    return process

def L1NtupleAODRAWCalo(process):

    L1NtupleRAWCalo(process)
    L1NtupleAODCalo(process)

    return process

def L1NtupleAODRAWEMULegacy(process):

    L1NtupleRAW(process)
    L1NtupleEMU(process)
    L1NtupleEMULegacy(process)
    L1NtupleAOD(process)

    return process


def L1NtupleAODRAWEMU(process):

    L1NtupleRAW(process)
    L1NtupleEMU(process)
    L1NtupleAOD(process)

    return process

def L1NtupleAODRAWEMUCalo(process):

    L1NtupleRAWCalo(process)
    L1NtupleEMUCalo(process)
    L1NtupleAODCalo(process)

    return process

def L1NtupleAODEMU(process):

    L1NtupleEMU(process)
    L1NtupleAOD(process)

    return process

def L1NtupleAODEMUCalo(process):

    L1NtupleEMUCalo(process)
    L1NtupleAODCalo(process)

    return process


def L1NtupleAODEMU_MC(process):

    L1NtupleEMU(process)
    L1NtupleAOD_MC(process)

    return process


def L1NtupleRAWEMUGEN_MC(process):

    L1NtupleRAW(process)
    L1NtupleEMU(process)
    L1NtupleGEN(process)

    return process

def L1NtupleAODEMUGEN_MC(process):

    L1NtupleEMU(process)
    L1NtupleAOD_MC(process)
    L1NtupleGEN(process)

    return process

def L1NtupleAODRAWEMUGEN_MC(process):

    L1NtupleRAW(process)
    L1NtupleEMU(process)
    L1NtupleAOD_MC(process)
    L1NtupleGEN(process)

    return process

def L1NtupleEMUNoEventTree(process):

    L1NtupleTFileOut(process)

    process.load('L1Trigger.L1TNtuples.L1NtupleEMU_cff')
    process.L1NtupleEMU = cms.Sequence( process.l1CaloTowerEmuTree+process.l1UpgradeEmuTree+process.l1UpgradeTfMuonEmuTree )
    process.l1ntuplesim = cms.Path(
        process.L1NtupleEMU
    )
    process.schedule.append(process.l1ntuplesim)

    return process

def PrefireVetoFilter(process):

    process.load('EventFilter.L1TRawToDigi.triggerRulePrefireVetoFilter_cfi')

    if hasattr(process, 'l1ntupleraw'):
        process.l1ntupleraw.insert(0,process.triggerRulePrefireVetoFilter)
    if hasattr(process, 'l1ntupleemu'):
        process.l1ntupleemu.insert(0,process.triggerRulePrefireVetoFilter)
    if hasattr(process, 'l1ntupleaod'):
        process.l1ntupleaod.insert(0,process.triggerRulePrefireVetoFilter)
    if hasattr(process, 'l1ntuplerawcalo'):
        process.l1ntuplerawcalo.insert(0,process.triggerRulePrefireVetoFilter)
    if hasattr(process, 'l1ntupleemucalo'):
        process.l1ntupleemucalo.insert(0,process.triggerRulePrefireVetoFilter)
    if hasattr(process, 'l1ntupleaodcalo'):
        process.l1ntupleaodcalo.insert(0,process.triggerRulePrefireVetoFilter)

    return process
