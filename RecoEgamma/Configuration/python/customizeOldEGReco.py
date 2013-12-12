import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.Utilities as psu
from  PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceAnyInputTag as _replaceTags

def customizeOldEGReco(process):
    for path in process.paths:
        sequences = getattr(process,path)
        #for seq in path:
        _replaceTags(sequences,
                     cms.InputTag('gedGsfElectrons'),
                     cms.InputTag('gsfElectrons'),
                     skipLabelTest=True)
        _replaceTags(sequences,
                     cms.InputTag('gedGsfElectronCores'),
                     cms.InputTag('gsfElectronCores'),
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
    process.particleFlowBlock.SCBarrel = cms.InputTag('correctedHybridSuperClusters')
    process.particleFlowBlock.SCEndcap = cms.InputTag('correctedMulti5x5SuperClustersWithPreshower')
    #add in conversions
    ## for PF
    
    process.allConversionSequence += process.allConversionOldEGSequence
    process.pfConversions.conversionCollection = cms.InputTag('allConversionsOldEG')        
    #return to old EG-based conversions based reco::Photon
    process.ckfTracksFromConversions += process.ckfTracksFromOldEGConversions
    process.conversionTrackProducers += process.oldegConversionTrackProducers
    process.conversionTrackMergers += process.oldegConversionTrackMergers
    if hasattr(process,'conversionSequence'):
        process.conversionSequence += process.oldegConversionSequence
    process.photonSequence.remove(process.mustachePhotonSequence)
    process.particleFlowBlock.EGPhotons = cms.InputTag('photons')
    process.particleFlowBlock.PhotonSelectionCuts = cms.vdouble(1,10,2.0, 0.001, 4.2, 0.003, 2.2, 0.001, 0.05, 10.0, 0.50),
    #tell PFProducer to use old PF electron / PF photon code
    process.particleFlowTmp.useEGammaFilters = cms.bool(False)
    process.particleFlowTmp.usePFPhotons = cms.bool(True)
    process.particleFlowTmp.usePFElectrons = cms.bool(True)
    #re-route PF linker to use old EG collections
    process.particleFlow.GsfElectrons = cms.InputTag('gsfElectrons')
    process.particleFlow.Photons = cms.InputTag('pfPhotonTranslator:pfphot')
    return process



def _customize_DQM(process):
    return process


def _customize_Validation(process):
    _replaceTags(process.validation_step,
                 cms.InputTag('gedGsfElectrons'),
                 cms.InputTag('gsfElectrons'),
                 skipLabelTest=True)
    _replaceTags(process.validation_step,
                 cms.InputTag('gedGsfElectronCores'),
                 cms.InputTag('gsfElectronCores'),
                 skipLabelTest=True)
    #don't ask... just don't ask
    if hasattr(process,'HLTSusyExoValFastSim'):
        process.HLTSusyExoValFastSim.PlotMakerRecoInput.electrons = \
                                                 cms.string('gsfElectrons')
        for pset in process.HLTSusyExoValFastSim.reco_parametersets:
            pset.electrons = cms.string('gsfElectrons')
    if hasattr(process,'HLTSusyExoVal'):
        process.HLTSusyExoVal.PlotMakerRecoInput.electrons = \
                                                 cms.string('gsfElectrons')
        for pset in process.HLTSusyExoVal.reco_parametersets:
            pset.electrons = cms.string('gsfElectrons')
    if hasattr(process,'hltHiggsValidator'):
        process.hltHiggsValidator.H2tau.recElecLabel = \
                                                cms.string('gsfElectrons')
        process.hltHiggsValidator.HZZ.recElecLabel = \
                                                cms.string('gsfElectrons')
        process.hltHiggsValidator.HWW.recElecLabel = \
                                                cms.string('gsfElectrons')
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
    process.egammaHighLevelRecoPostPF.insert(process.gsfElectronMergingSequence,0)
    process.famosParticleFlowReco.insert(process.pfPhotonTranslatorSequence,7)
    process.famosParticleFlowReco.insert(process.pfElectronTranslatorSequence,7)

    process.interestingEcalDetIdEB.basicClustersLabel = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters")
    process.interestingEcalDetIdEE.basicClustersLabel = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapBasicClusters")
    process.reducedEcalRecHitsES. EndcapSuperClusterCollection = cms.InputTag('correctedMulti5x5SuperClustersWithPreshower')

    process.interestingEleIsoDetIdEB.emObjectLabel = cms.InputTag('gsfElectrons')
    process.interestingEleIsoDetIdEE.emObjectLabel = cms.InputTag('gsfElectrons')
    process.interestingGamIsoDetIdEB.emObjectLabel = cms.InputTag('photons')
    process.interestingGamIsoDetIdEE.emObjectLabel = cms.InputTag('photons')  
        
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
        cms.InputTag("pfElectronInterestingEcalDetIdEB"),
        cms.InputTag("pfPhotonInterestingEcalDetIdEB"),
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
        cms.InputTag("pfElectronInterestingEcalDetIdEE"),
        cms.InputTag("pfPhotonInterestingEcalDetIdEE"),
        # muons
        cms.InputTag("muonEcalDetIds"),
        # high pt tracks
        cms.InputTag("interestingTrackEcalDetIds")
        )
    process.allConversionsOldEG.src = cms.InputTag('gsfGeneralConversionTrackMerger')

    if hasattr(process,'ecalDrivenElectronSeeds'):
        process.ecalDrivenElectronSeeds.barrelSuperClusters = cms.InputTag('correctedHybridSuperClusters')
        process.ecalDrivenElectronSeeds.endcapSuperClusters = cms.InputTag('correctedMulti5x5SuperClustersWithPreshower')
    return process


def _customize_Reco(process):
    process=_configurePFForGEDEGamma(process)    
    process.egammaHighLevelRecoPostPF.insert(process.gsfElectronMergingSequence,0)
    process.particleFlowReco.insert(process.pfPhotonTranslatorSequence,7)
    process.particleFlowReco.insert(process.pfElectronTranslatorSequence,7)

    process.interestingEcalDetIdEB.basicClustersLabel = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters")
    process.interestingEcalDetIdEE.basicClustersLabel = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapBasicClusters")
    process.reducedEcalRecHitsES. EndcapSuperClusterCollection = cms.InputTag('correctedMulti5x5SuperClustersWithPreshower')

    process.interestingEleIsoDetIdEB.emObjectLabel = cms.InputTag('gsfElectrons')
    process.interestingEleIsoDetIdEE.emObjectLabel = cms.InputTag('gsfElectrons')
    process.interestingGamIsoDetIdEB.emObjectLabel = cms.InputTag('photons')
    process.interestingGamIsoDetIdEE.emObjectLabel = cms.InputTag('photons')    
    
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
        cms.InputTag("pfElectronInterestingEcalDetIdEB"),
        cms.InputTag("pfPhotonInterestingEcalDetIdEB"),
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
        cms.InputTag("pfElectronInterestingEcalDetIdEE"),
        cms.InputTag("pfPhotonInterestingEcalDetIdEE"),
        # muons
        cms.InputTag("muonEcalDetIds"),
        # high pt tracks
        cms.InputTag("interestingTrackEcalDetIds")
        )

    if hasattr(process,'ecalDrivenElectronSeeds'):
        process.ecalDrivenElectronSeeds.barrelSuperClusters = cms.InputTag('correctedHybridSuperClusters')
        process.ecalDrivenElectronSeeds.endcapSuperClusters = cms.InputTag('correctedMulti5x5SuperClustersWithPreshower')
    return process


def _customize_harvesting(process):
    if hasattr(process,'oldpfPhotonPostprocessing'):
        process.photonPostProcessor += process.oldpfPhotonPostprocessing
    return process
