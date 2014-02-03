import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.Utilities as psu
from  PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceAnyInputTag as _replaceTags

def customizePFforEGammaGED(process):
    for path in process.paths:
        sequences = getattr(process,path)
        #for seq in path:
        _replaceTags(sequences,
                     cms.InputTag('gsfElectrons'),
                     cms.InputTag('gedGsfElectrons'),
                     skipLabelTest=True)
        _replaceTags(sequences,
                     cms.InputTag('gsfElectronCores'),
                     cms.InputTag('gedGsfElectronCores'),
                     skipLabelTest=True)

    # all the rest:
    if hasattr(process,'DigiToRaw'):
        process=_customize_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=_customize_RawToDigi(process)
    if hasattr(process,'reconstruction'):
        process=_customize_Reco(process)
    if hasattr(process,'reconstructionWithFamos'):
        process=_customize_FastSim(process)
    if hasattr(process,'digitisation_step'):
        process=_customize_Digi(process)
    if hasattr(process,'HLTSchedule'):
        process=_customize_HLT(process)
    if hasattr(process,'L1simulation_step'):
        process=_customize_L1Emulator(process)
    if hasattr(process,'dqmoffline_step'):
        process=_customize_DQM(process)
    if hasattr(process,'dqmHarvesting'):
        process=_customize_harvesting(process)
    if hasattr(process,'validation_step'):
        process=_customize_Validation(process)


    return process

def _configurePFForGEDEGamma(process):  
    #for later
    process.particleFlowBlock.SCBarrel = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel')
    process.particleFlowBlock.SCEndcap = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower')
    #add in conversions
    ## for PF
    
    process.allConversionSequence += process.allConversionMustacheSequence
    process.pfConversions.conversionCollection = cms.InputTag('allConversionsMustache')        
    #setup mustache based reco::Photon
    process.ckfTracksFromConversions += process.ckfTracksFromMustacheConversions
    process.conversionTrackProducers += process.mustacheConversionTrackProducers
    process.conversionTrackMergers += process.mustacheConversionTrackMergers
    if hasattr(process,'conversionSequence'):
        process.conversionSequence += process.mustacheConversionSequence
    process.photonSequence += process.mustachePhotonSequence
    process.particleFlowBlock.EGPhotons = cms.InputTag('mustachePhotons')
    #tell PFProducer to use PFEG objects / gedTmp
    process.particleFlowTmp.useEGammaFilters = cms.bool(True)
    process.particleFlowTmp.usePFPhotons = cms.bool(False)
    process.particleFlowTmp.usePFElectrons = cms.bool(False)
    #re-route PF linker to use ged collections
    process.particleFlow.GsfElectrons = cms.InputTag('gedGsfElectrons')
    process.particleFlow.Photons = cms.InputTag('gedPhotons')
    return process



def _customize_DQM(process):
    return process


def _customize_Validation(process):
    _replaceTags(process.validation_step,
                 cms.InputTag('gsfElectrons'),
                 cms.InputTag('gedGsfElectrons'),
                 skipLabelTest=True)
    _replaceTags(process.validation_step,
                 cms.InputTag('gsfElectronCores'),
                 cms.InputTag('gedGsfElectronCores'),
                 skipLabelTest=True)
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


def _customize_Digi(process):
    return process


def _customize_L1Emulator(process):
    return process


def _customize_RawToDigi(process):
    return process


def _customize_DigiToRaw(process):
    return process


def _customize_HLT(process):
    return process

def _customize_FastSim(process):
    process=_configurePFForGEDEGamma(process)
    process.famosParticleFlowSequence.remove(process.pfElectronTranslatorSequence)
    process.famosParticleFlowSequence.remove(process.pfPhotonTranslatorSequence)
    process.egammaHighLevelRecoPostPF.remove(process.gsfElectronMergingSequence)
    process.reducedEcalRecHitsEB.interestingDetIdCollections = cms.VInputTag(
        # ecal
        cms.InputTag("interestingEcalDetIdEB"),
        cms.InputTag("interestingEcalDetIdEBU"),
        # egamma
        cms.InputTag("interestingEleIsoDetIdEB"),
        cms.InputTag("interestingGamIsoDetIdEB"),
        # tau
        #cms.InputTag("caloRecoTauProducer"),
        #pf
        #cms.InputTag("pfElectronInterestingEcalDetIdEB"),
        #cms.InputTag("pfPhotonInterestingEcalDetIdEB"),
        # muons
        cms.InputTag("muonEcalDetIds"),
        # high pt tracks
        cms.InputTag("interestingTrackEcalDetIds")
        )
    process.reducedEcalRecHitsEE.interestingDetIdCollections = cms.VInputTag(
        # ecal
        cms.InputTag("interestingEcalDetIdEE"),
        # egamma
        cms.InputTag("interestingEleIsoDetIdEE"),
        cms.InputTag("interestingGamIsoDetIdEE"),
        # tau
        #cms.InputTag("caloRecoTauProducer"),
        #pf
        #cms.InputTag("pfElectronInterestingEcalDetIdEE"),
        #cms.InputTag("pfPhotonInterestingEcalDetIdEE"),
        # muons
        cms.InputTag("muonEcalDetIds"),
        # high pt tracks
        cms.InputTag("interestingTrackEcalDetIds")
        )
    process.allConversionsMustache.src = cms.InputTag('gsfGeneralConversionTrackMerger')
    if hasattr(process,'ecalDrivenElectronSeeds'):
        process.ecalDrivenElectronSeeds.barrelSuperClusters = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel')
        process.ecalDrivenElectronSeeds.endcapSuperClusters = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower')
    return process


def _customize_Reco(process):
    process=_configurePFForGEDEGamma(process)
    process.particleFlowReco.remove(process.pfElectronTranslatorSequence)
    process.particleFlowReco.remove(process.pfPhotonTranslatorSequence)
    process.egammaHighLevelRecoPostPF.remove(process.gsfElectronMergingSequence)
    process.reducedEcalRecHitsEB.interestingDetIdCollections = cms.VInputTag(
        # ecal
        cms.InputTag("interestingEcalDetIdEB"),
        cms.InputTag("interestingEcalDetIdEBU"),
        # egamma
        cms.InputTag("interestingEleIsoDetIdEB"),
        cms.InputTag("interestingGamIsoDetIdEB"),
        # tau
        #cms.InputTag("caloRecoTauProducer"),
        #pf
        #cms.InputTag("pfElectronInterestingEcalDetIdEB"),
        #cms.InputTag("pfPhotonInterestingEcalDetIdEB"),
        # muons
        cms.InputTag("muonEcalDetIds"),
        # high pt tracks
        cms.InputTag("interestingTrackEcalDetIds")
        )
    process.reducedEcalRecHitsEE.interestingDetIdCollections = cms.VInputTag(
        # ecal
        cms.InputTag("interestingEcalDetIdEE"),
        # egamma
        cms.InputTag("interestingEleIsoDetIdEE"),
        cms.InputTag("interestingGamIsoDetIdEE"),
        # tau
        #cms.InputTag("caloRecoTauProducer"),
        #pf
        #cms.InputTag("pfElectronInterestingEcalDetIdEE"),
        #cms.InputTag("pfPhotonInterestingEcalDetIdEE"),
        # muons
        cms.InputTag("muonEcalDetIds"),
        # high pt tracks
        cms.InputTag("interestingTrackEcalDetIds")
        )

    if hasattr(process,'ecalDrivenElectronSeeds'):
        process.ecalDrivenElectronSeeds.barrelSuperClusters = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel')
        process.ecalDrivenElectronSeeds.endcapSuperClusters = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower')
    return process


def _customize_harvesting(process):
    if hasattr(process,'oldpfPhotonPostprocessing'):
        process.photonPostProcessor.remove(process.oldpfPhotonPostprocessing)
    return process
