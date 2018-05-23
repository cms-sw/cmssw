import FWCore.ParameterSet.Config as cms

import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
import RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi as recoparam
import RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi as recocalibparam 
import hgcalLayersCalibrationCoefficients_cfi as layercalibparam


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
    HGCDigisEE = cms.InputTag('mix:HGCDigisEE'),
    HGCDigisFH = cms.InputTag('mix:HGCDigisHEfront'),
    HGCDigisBH = cms.InputTag('mix:HGCDigisHEback'),
    eeSimHits = cms.InputTag('g4SimHits:HGCHitsEE'),
    fhSimHits = cms.InputTag('g4SimHits:HGCHitsHEfront'),
    bhSimHits = cms.InputTag('g4SimHits:HcalHits'),
    isSimhitComp = cms.bool(False)
)

ntuple_triggercells = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCTriggerCells'),
    TriggerCells = cms.InputTag('hgcalTriggerPrimitiveDigiProducer:calibratedTriggerCells'),
    Multiclusters = cms.InputTag('hgcalTriggerPrimitiveDigiProducer:cluster3D'),
    eeSimHits = cms.InputTag('g4SimHits:HGCHitsEE'),
    fhSimHits = cms.InputTag('g4SimHits:HGCHitsHEfront'),
    bhSimHits = cms.InputTag('g4SimHits:HcalHits'),
    FillSimEnergy = cms.bool(False),
    fcPerMip = fcPerMip,
    keV2fC = keV2fC,
    layerWeights = layerWeights,
    thicknessCorrections = thicknessCorrections,
    FilterCellsInMulticlusters = cms.bool(True)
)

ntuple_clusters = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCClusters'),
    Clusters = cms.InputTag('hgcalTriggerPrimitiveDigiProducer:cluster2D'),
    Multiclusters = cms.InputTag('hgcalTriggerPrimitiveDigiProducer:cluster3D'),
    FilterClustersInMulticlusters = cms.bool(True)
)

from L1Trigger.L1THGCal.egammaIdentification import egamma_identification_drnn_cone
ntuple_multicluster = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCMulticlusters'),
    Multiclusters = cms.InputTag('hgcalTriggerPrimitiveDigiProducer:cluster3D'),
    EGIdentification = egamma_identification_drnn_cone.clone()
)

ntuple_panels = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCPanels'),
    TriggerCells = cms.InputTag('hgcalTriggerPrimitiveDigiProducer:calibratedTriggerCells')
)

ntuple_tower = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCTowers'),
    Towers = cms.InputTag('hgcalTriggerPrimitiveDigiProducer:tower')
)

hgcalTriggerNtuplizer = cms.EDAnalyzer(
    "HGCalTriggerNtupleManager",
    Ntuples = cms.VPSet(
        ntuple_event,
        ntuple_gen,
        ntuple_genjet,
        ntuple_gentau,
        ntuple_digis,
        ntuple_triggercells,
        ntuple_clusters,
        ntuple_multicluster,
        ntuple_tower
    )
)
