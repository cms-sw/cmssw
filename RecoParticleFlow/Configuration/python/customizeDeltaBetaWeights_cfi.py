'''Customization functions for cmsDriver to get neutral weighted isolation'''
import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *


def customize(process):
    '''run neutral particle weighting sequence and use it for isolation of electrons, muons and photons

       syntax: --customise RecoParticleFlow/Configuration/customizeDeltaBetaWeights_cfi.customize
       It will add 2 new sequences to the RECO sequence that will produce pfWeightedPhotons and 
       pfWeightedNeutralHadrons. They are produced from pfAllPhotons and pfAllNeutralHadrons by rescaling
       pt of each particle by a weight that reflects the probability that it is from pileup. The formula is
       w = sumNPU/(sumNPU+sumPU). The sums are running over all charged particles from the PV (NPU) or from the PU.
       The function used in the sum is ln(pt(i)/deltaR(i,j)) where i is neutral particle that is being weighted and j
       is the charged particle (either PU or NPU) that is used to access 'pileupility' of a particle.

       Neutral isolation of electrons, muons and photons is calculated using the weighed collection.
    '''

    if hasattr(process,'pfParticleSelectionSequence'): 
        process.load("CommonTools.ParticleFlow.deltaBetaWeights_cff")
        process.pfParticleSelectionSequence += process.pfDeltaBetaWeightingSequence

    if hasattr(process,'elPFIsoDepositNeutral'): 
        process.elPFIsoDepositNeutral=isoDepositReplace('pfElectronTranslator:pf','pfWeightedNeutralHadrons')

    if hasattr(process,'elPFIsoDepositGamma'):
        process.elPFIsoDepositGamma=isoDepositReplace('pfElectronTranslator:pf','pfWeightedPhotons')

    if hasattr(process,'gedElPFIsoDepositNeutral'):
        process.gedElPFIsoDepositNeutral=isoDepositReplace('gedGsfElectronsTmp','pfWeightedNeutralHadrons')

    if hasattr(process,'gedElPFIsoDepositGamma'):
        process.gedElPFIsoDepositGamma=isoDepositReplace('gedGsfElectronsTmp','pfWeightedPhotons')

    if hasattr(process,'muPFIsoDepositNeutral'):
       process.muPFIsoDepositNeutral=isoDepositReplace('muons1stStep','pfWeightedNeutralHadrons')

    if hasattr(process,'muPFIsoDepositGamma'):
        process.muPFIsoDepositGamma=isoDepositReplace('muons1stStep','pfWeightedPhotons')

    if hasattr(process,'phPFIsoDepositNeutral'):
       process.phPFIsoDepositNeutral=isoDepositReplace('pfSelectedPhotons','pfWeightedNeutralHadrons')

    if hasattr(process,'phPFIsoDepositGamma'):
        process.phPFIsoDepositGamma.ExtractorPSet.inputCandView = cms.InputTag("pfWeightedPhotons")

    return process


def customizeElectronsOnly(process):
    '''run neutral particle weighting sequence and use it for isolation of electrons only.

    syntax: --customise RecoParticleFlow/Configuration/customizeDeltaBetaWeights_cfi.customizeElectronsOnly
    Same as customize, only that the weighted collections are used only for electron neutral isolation, 
    while muons and photons are left untouched.
    '''

    if hasattr(process,'pfParticleSelectionSequence'): 
        process.load("CommonTools.ParticleFlow.deltaBetaWeights_cff")
        process.pfParticleSelectionSequence += process.pfDeltaBetaWeightingSequence

    if hasattr(process,'elPFIsoDepositNeutral'):
        process.elPFIsoDepositNeutral=isoDepositReplace('pfElectronTranslator:pf','pfWeightedNeutralHadrons')

    if hasattr(process,'elPFIsoDepositGamma'):
        process.elPFIsoDepositGamma=isoDepositReplace('pfElectronTranslator:pf','pfWeightedPhotons')

    if hasattr(process,'gedElPFIsoDepositNeutral'):
        process.gedElPFIsoDepositNeutral=isoDepositReplace('gedGsfElectronsTmp','pfWeightedNeutralHadrons')

    if hasattr(process,'gedElPFIsoDepositGamma'):
        process.gedElPFIsoDepositGamma=isoDepositReplace('gedGsfElectronsTmp','pfWeightedPhotons')

    return process


def customizeMuonsOnly(process):
    '''run neutral particle weighting sequence and use it for isolation of muonss only.

    syntax: --customise RecoParticleFlow/Configuration/customizeDeltaBetaWeights_cfi.customizeMuonsOnly
    Same as customize, only that the weighted collections are used only for muon neutral isolation,
    while electronss and photons are left untouched.
    '''

    if hasattr(process,'pfParticleSelectionSequence'): 
        process.load("CommonTools.ParticleFlow.deltaBetaWeights_cff")
        process.pfParticleSelectionSequence += process.pfDeltaBetaWeightingSequence

    if hasattr(process,'muPFIsoDepositNeutral'):
       process.muPFIsoDepositNeutral=isoDepositReplace('muons1stStep','pfWeightedNeutralHadrons')

    if hasattr(process,'muPFIsoDepositGamma'):
        process.muPFIsoDepositGamma=isoDepositReplace('muons1stStep','pfWeightedPhotons')

    return process


def customizePhotonsOnly(process):
    '''run neutral particle weighting sequence and use it for isolation of muons only.

    syntax: --customise RecoParticleFlow/Configuration/customizeDeltaBetaWeights_cfi.customizePhotonsOnly
    Same as customize, only that the weighted collections are used only for photon neutral isolation,
    while electronss and muons are left untouched.
    ''' 

    if hasattr(process,'pfParticleSelectionSequence'): 
        process.load("CommonTools.ParticleFlow.deltaBetaWeights_cff")
        process.pfParticleSelectionSequence += process.pfDeltaBetaWeightingSequence

    if hasattr(process,'phPFIsoDepositNeutral'):
       process.phPFIsoDepositNeutral=isoDepositReplace('pfSelectedPhotons','pfWeightedNeutralHadrons')

    if hasattr(process,'phPFIsoDepositGamma'):
        process.phPFIsoDepositGamma.ExtractorPSet.inputCandView = cms.InputTag("pfWeightedPhotons")


    return process
