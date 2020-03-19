import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCalUtilities.caloTruthCellsProducer_cfi import caloTruthCellsProducer
from L1Trigger.L1THGCalUtilities.hgcalTriggerNtuples_cfi import *

ntuple_multiclusters_fulltruth = ntuple_multiclusters.clone(
    Multiclusters = cms.InputTag('caloTruthCellsProducer'),
    Prefix = cms.untracked.string('cl3dfulltruth')
)
hgcalTriggerNtuplizer.Ntuples.append(ntuple_multiclusters_fulltruth)

# If caloTruthCellsProducer.makeCellsCollection is True, can run the clustering algorithm over output cells too

if caloTruthCellsProducer.makeCellsCollection:
    ## ntuplize clusters and towers

    ntuple_triggercells.caloParticlesToCells = cms.InputTag('caloTruthCellsProducer')
    ntuple_triggercells.FillTruthMap = cms.bool(True)
    
    ntuple_triggercells_truth = ntuple_triggercells.clone(
        TriggerCells = cms.InputTag('caloTruthCellsProducer'),
        Multiclusters = cms.InputTag('hgcalTruthBackEndLayer2Producer:HGCalBackendLayer2Processor3DClustering'),
        Prefix = cms.untracked.string('tctruth'),
        FillTruthMap = cms.bool(False)
    )
    
    ntuple_clusters_truth = ntuple_clusters.clone(
        Clusters = cms.InputTag('hgcalTruthBackEndLayer1Producer:HGCalBackendLayer1Processor2DClustering'),
        Prefix = cms.untracked.string('cltruth')
    )
    
    ntuple_multiclusters_truth = ntuple_multiclusters.clone(
        Multiclusters = cms.InputTag('hgcalTruthBackEndLayer2Producer:HGCalBackendLayer2Processor3DClustering'),
        Prefix = cms.untracked.string('cl3dtruth')
    )
    
    ntuple_towers_truth = ntuple_towers.clone(
        Towers = cms.InputTag('hgcalTruthTowerProducer:HGCalTowerProcessor'),
        Prefix = cms.untracked.string('towertruth')
    )

    hgcalTriggerNtuplizer.Ntuples.append(ntuple_triggercells_truth)
    hgcalTriggerNtuplizer.Ntuples.append(ntuple_multiclusters_truth)
    hgcalTriggerNtuplizer.Ntuples.append(ntuple_towers_truth)
