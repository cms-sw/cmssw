import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *


def customize(process):

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

    if hasattr(process,'pfParticleSelectionSequence'): 
        process.load("CommonTools.ParticleFlow.deltaBetaWeights_cff")
        process.pfParticleSelectionSequence += process.pfDeltaBetaWeightingSequence

    if hasattr(process,'muPFIsoDepositNeutral'):
       process.muPFIsoDepositNeutral=isoDepositReplace('muons1stStep','pfWeightedNeutralHadrons')

    if hasattr(process,'muPFIsoDepositGamma'):
        process.muPFIsoDepositGamma=isoDepositReplace('muons1stStep','pfWeightedPhotons')

    return process


def customizePhotonsOnly(process):

    if hasattr(process,'pfParticleSelectionSequence'): 
        process.load("CommonTools.ParticleFlow.deltaBetaWeights_cff")
        process.pfParticleSelectionSequence += process.pfDeltaBetaWeightingSequence

    if hasattr(process,'phPFIsoDepositNeutral'):
       process.phPFIsoDepositNeutral=isoDepositReplace('pfSelectedPhotons','pfWeightedNeutralHadrons')

    if hasattr(process,'phPFIsoDepositGamma'):
        process.phPFIsoDepositGamma.ExtractorPSet.inputCandView = cms.InputTag("pfWeightedPhotons")


    return process
