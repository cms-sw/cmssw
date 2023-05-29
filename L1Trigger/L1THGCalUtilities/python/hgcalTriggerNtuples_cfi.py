from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms

import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
import RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi as recoparam
import RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi as recocalibparam
import L1Trigger.L1THGCal.hgcalLayersCalibrationCoefficients_cfi as layercalibparam


fcPerMip = recoparam.HGCalUncalibRecHit.HGCEEConfig.fCPerMIP
keV2fC = digiparam.hgceeDigitizer.digiCfg.keV2fC
layerWeights = layercalibparam.TrgLayer_dEdX_weights
thicknessCorrections = recocalibparam.HGCalRecHit.thicknessCorrection

ntuple_event = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleEvent')
)

from FastSimulation.Event.ParticleFilter_cfi import ParticleFilterBlock
PartFilterConfig = ParticleFilterBlock.ParticleFilter.copy()
PartFilterConfig.protonEMin = cms.double(100000)
PartFilterConfig.etaMax = cms.double(3.1)

ntuple_gen = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleGen'),
    GenParticles = cms.InputTag('genParticles'),
    GenPU = cms.InputTag('addPileupInfo'),
    MCEvent = cms.InputTag('generatorSmeared'),
    SimTracks = cms.InputTag('g4SimHits'),
    SimVertices = cms.InputTag('g4SimHits'),
    particleFilter = PartFilterConfig
)

ntuple_gentau = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleGenTau'),
    GenParticles = cms.InputTag('genParticles'),
    isPythia8 = cms.bool(False)
)

ntuple_genjet = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleGenJet'),
    GenJets = cms.InputTag('ak4GenJetsNoNu')
)

ntuple_digis = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCDigis'),
    HGCDigisEE = cms.InputTag('simHGCalUnsuppressedDigis:EE'),
    HGCDigisFH = cms.InputTag('simHGCalUnsuppressedDigis:HEfront'),
    HGCDigisBH = cms.InputTag('simHGCalUnsuppressedDigis:HEback'),
    eeSimHits = cms.InputTag('g4SimHits:HGCHitsEE'),
    fhSimHits = cms.InputTag('g4SimHits:HGCHitsHEfront'),
    bhSimHits = cms.InputTag('g4SimHits:HGCHitsHEback'),
    isSimhitComp = cms.bool(False),
    digiBXselect = cms.vuint32(2)
)

ntuple_triggercells = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCTriggerCells'),
    TriggerCells = cms.InputTag('l1tHGCalConcentratorProducer:HGCalConcentratorProcessorSelection'),
    Multiclusters = cms.InputTag('l1tHGCalBackEndLayer2Producer:HGCalBackendLayer2Processor3DClustering'),
    eeSimHits = cms.InputTag('g4SimHits:HGCHitsEE'),
    fhSimHits = cms.InputTag('g4SimHits:HGCHitsHEfront'),
    bhSimHits = cms.InputTag('g4SimHits:HGCHitsHEback'), 
    FillSimEnergy = cms.bool(False),
    FillTruthMap = cms.bool(False),
    fcPerMip = fcPerMip,
    keV2fC = keV2fC,
    layerWeights = layerWeights,
    thicknessCorrections = thicknessCorrections,
    FilterCellsInMulticlusters = cms.bool(False)
)

ntuple_triggersums = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCTriggerSums'),
    TriggerSums = cms.InputTag('l1tHGCalConcentratorProducer:HGCalConcentratorProcessorSelection'),
)

ntuple_econdata = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCConcentratorData'),
    ConcentratorData = cms.InputTag('l1tHGCalConcentratorProducer:HGCalConcentratorProcessorSelection'),
)

ntuple_clusters = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCClusters'),
    Clusters = cms.InputTag('l1tHGCalBackEndLayer1Producer:HGCalBackendLayer1Processor2DClustering'),
    Multiclusters = cms.InputTag('l1tHGCalBackEndLayer2Producer:HGCalBackendLayer2Processor3DClustering'),
    FilterClustersInMulticlusters = cms.bool(False)
)

from L1Trigger.L1THGCal.egammaIdentification import egamma_identification_histomax
ntuple_multiclusters = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCMulticlusters'),
    Multiclusters = cms.InputTag('l1tHGCalBackEndLayer2Producer:HGCalBackendLayer2Processor3DClustering'),
    EGIdentification = egamma_identification_histomax.clone(),
    FillLayerInfo = cms.bool(False),
    FillInterpretationInfo = cms.bool(True)
)

ntuple_towers = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCTowers'),
    Towers = cms.InputTag('l1tHGCalTowerProducer:HGCalTowerProcessor')
)

l1tHGCalTriggerNtuplizer = cms.EDAnalyzer(
    "HGCalTriggerNtupleManager",
    Ntuples = cms.VPSet(
        ntuple_event,
        ntuple_gen,
        ntuple_genjet,
        ntuple_gentau,
        ntuple_digis,
        ntuple_triggercells,
        ntuple_triggersums,
        ntuple_multiclusters,
        ntuple_towers
    )
)
