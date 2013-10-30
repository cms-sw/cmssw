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
    return process



def customize_DQM(process):
    replaceTags(process.dqmoffline_step,
                cms.InputTag('gsfElectrons'),
                cms.InputTag('gedGsfElectrons'))
    replaceTags(process.dqmoffline_step,
                cms.InputTag('gsfElectronCores'),
                cms.InputTag('gedGsfElectronCores'))
    return process


def customize_Validation(process):
    replaceTags(process.validation_step,
                cms.InputTag('gsfElectrons'),
                cms.InputTag('gedGsfElectrons'))
    replaceTags(process.validation_step,
                cms.InputTag('gsfElectronCores'),
                cms.InputTag('gedGsfElectronCores'))
    #don't ask... just don't ask
    if hasattr(process,'HLTSusyExoValFastSim'):
        process.HLTSusyExoValFastSim.PlotMakerRecoInput.electrons = \
                                                 cms.string('gedGsfElectrons')
        for pset in process.HLTSusyExoValFastSim.reco_parametersets:
            pset.electrons = cms.string('gedGsfElectrons')
    if hasattr(process,'HLTSusyExoVal'):
        process.HLTSusyExoVal.PlotMakerRecoInput.electrons = \
                                                 cms.string('gedGsfElectrons')
        for pset in process.HLTSusyExoVal.reco_parametersets:
            pset.electrons = cms.string('gedGsfElectrons')
    if hasattr(process,'hltHiggsValidator'):
        process.hltHiggsValidator.H2tau.recElecLabel = \
                                                cms.string('gedGsfElectrons')
        process.hltHiggsValidator.HZZ.recElecLabel = \
                                                cms.string('gedGsfElectrons')
        process.hltHiggsValidator.HWW.recElecLabel = \
                                                cms.string('gedGsfElectrons')
    if hasattr(process,'oldpfPhotonValidation'):
        process.photonValidationSequence.remove(process.oldpfPhotonValidation)
    return process


def customize_Digi(process):
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
    process.famosParticleFlowSequence.remove(process.pfElectronTranslatorSequence)
    process.famosParticleFlowSequence.remove(process.pfPhotonTranslatorSequence)
    process.egammaHighLevelRecoPostPF.remove(process.gsfElectronMergingSequence)
    replaceTags(process.reconstructionWithFamos,
                cms.InputTag('gsfElectrons'),
                cms.InputTag('gedGsfElectrons'))
    if hasattr(process,'ecalDrivenElectronSeeds'):
        process.ecalDrivenElectronSeeds.barrelSuperClusters = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel')
        process.ecalDrivenElectronSeeds.endcapSuperClusters = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower')
    return process


def customize_Reco(process):
    process=configurePFForGEDEGamma(process)
    process.particleFlowReco.remove(process.pfElectronTranslatorSequence)
    process.particleFlowReco.remove(process.pfPhotonTranslatorSequence)
    process.egammaHighLevelRecoPostPF.remove(process.gsfElectronMergingSequence)
    replaceTags(process.reconstruction,
                cms.InputTag('gsfElectrons'),
                cms.InputTag('gedGsfElectrons'))
    if hasattr(process,'ecalDrivenElectronSeeds'):
        process.ecalDrivenElectronSeeds.barrelSuperClusters = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel')
        process.ecalDrivenElectronSeeds.endcapSuperClusters = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower')
    return process


def customize_harvesting(process):
    if hasattr(process,'oldpfPhotonPostprocessing'):
        process.photonPostProcessor.remove(process.oldpfPhotonPostprocessing)
    return process

def recoOutputCustoms(process):
    return process
