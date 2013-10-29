import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.Utilities as psu
from  PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceAnyInputTag as replaceTags

def customizePFforEGammaGED(process):
    
    # all the rest:
    if hasattr(process,'DigiToRaw'):
        process=customize_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customize_RawToDigi(process)
    if hasattr(process,'reconstruction'):
        process=customize_Reco(process)
    if hasattr(process,'reconstructionWithFamos'):
        process=customize_FastSim(process)
    if hasattr(process,'digitisation_step'):
        process=customize_Digi(process)
    if hasattr(process,'HLTSchedule'):
        process=customize_HLT(process)
    if hasattr(process,'L1simulation_step'):
        process=customize_L1Emulator(process)
    if hasattr(process,'dqmoffline_step'):
        process=customize_DQM(process)
    if hasattr(process,'dqmHarvesting'):
        process=customize_harvesting(process)
    if hasattr(process,'validation_step'):
        process=customize_Validation(process)

    return process

def configurePFForGEDEGamma(process):
    process.particleFlowTmp.useEGammaFilters = cms.bool(True)
    process.particleFlowTmp.usePFPhotons = cms.bool(False)
    process.particleFlowTmp.usePFElectrons = cms.bool(False)
    process.particleFlow.GsfElectrons = cms.InputTag('gedGsfElectrons')
    process.particleFlow.Photons = cms.InputTag('gedPhotons')
    process.particleFlowReco.remove(the_process.pfElectronTranslatorSequence)
    process.particleFlowReco.remove(the_process.pfPhotonTranslatorSequence)
    process.egammaHighLevelRecoPostPF.remove(the_process.gsfElectronMergingSequence)
    return process



def customize_DQM(process):
    replaceTags(process.dqmoffline_step,
                cms.InputTag('gsfElectrons'),
                cms.InputTag('gedGsfElectrons'))
    return process


def customize_Validation(process):
    replaceTags(process.validation_step,
                cms.InputTag('gsfElectrons'),
                cms.InputTag('gedGsfElectrons'))
    return process


def customize_Digi(process):
    process=digiEventContent(process)
    return process


def customize_L1Emulator(process):
    return process


def customize_RawToDigi(process):
    return process


def customize_DigiToRaw(process):
    return process


def customize_HLT(process):
    return process

def customize_FastSim(process):    
    process=configurePFForGEDEGamma(process)
    replaceTags(process.reconstructionWithFamos,
                cms.InputTag('gsfElectrons'),
                cms.InputTag('gedGsfElectrons'))    
    return process


def customize_Reco(process):
    process=configurePFForGEDEGamma(process)
    replaceTags(process.reconstruction,
                cms.InputTag('gsfElectrons'),
                cms.InputTag('gedGsfElectrons'))    
    return process


def customize_harvesting(process):
    replaceTags(process.dqmHarvesting,
                cms.InputTag('gsfElectrons'),
                cms.InputTag('gedGsfElectrons'))
    return process

def recoOutputCustoms(process):
    return process
