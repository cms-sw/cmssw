import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
from RecoTauTag.RecoTau.PFRecoTauPFJetInputs_cfi import PFRecoTauPFJetInputs
'''

Configuration for 'shrinkingCone' PFTau Producer

See PFT-08-001 for a description of the algorithm.

'''

_shrinkingConeRecoTausConfig = cms.PSet(
    name = cms.string("shrinkingCone"),
    qualityCuts = PFTauQualityCuts,
    # If true, consider PFLeptons (e/mu) as charged hadrons.
    usePFLeptons = cms.bool(True),
    pfCandSrc = cms.InputTag("particleFlow"),
    plugin = cms.string("RecoTauBuilderConePlugin"),
    leadObjectPt = cms.double(5.0),
    matchingCone = cms.string('0.3'),
    signalConeChargedHadrons = cms.string('min(max(5.0/et(), 0.07), 0.15)'),
    isoConeChargedHadrons = cms.string('0.4'),
    signalConePiZeros = cms.string('0.15'),
    isoConePiZeros = cms.string('0.4'),
    signalConeNeutralHadrons = cms.string('0.15'),
    isoConeNeutralHadrons = cms.string('0.4'),
    maxSignalConeChargedHadrons = cms.int32(-1) # CV: upper limit on number of signalConeChargedHadrons disabled per default
)

shrinkingConeRecoTaus = cms.EDProducer(
    "RecoTauProducer",
    jetSrc = PFRecoTauPFJetInputs.inputJetCollection,
    piZeroSrc = cms.InputTag("ak5PFJetsRecoTauPiZeros"),
    jetRegionSrc = cms.InputTag("recoTauAK5PFJets08Region"),
    builders = cms.VPSet(
        _shrinkingConeRecoTausConfig
    ),
    # Build an empty tau in the case that a jet does not have any tracks
    buildNullTaus = cms.bool(True),
    modifiers = cms.VPSet(
        # Electron rejection
        cms.PSet(
            name = cms.string("shrinkingConeElectronRej"),
            plugin = cms.string("RecoTauElectronRejectionPlugin"),
            #Electron rejection parameters
            ElectronPreIDProducer                = cms.InputTag("elecpreid"),
            EcalStripSumE_deltaPhiOverQ_minValue = cms.double(-0.1),
            EcalStripSumE_deltaPhiOverQ_maxValue = cms.double(0.5),
            EcalStripSumE_minClusEnergy          = cms.double(0.1),
            EcalStripSumE_deltaEta               = cms.double(0.03),
            ElecPreIDLeadTkMatch_maxDR           = cms.double(0.01),
            maximumForElectrionPreIDOutput       = cms.double(-0.1),
            DataType = cms.string("AOD"),
        )
    )
)
