import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCalUtilities.caloTruthCellsProducer_cfi import l1tCaloTruthCellsProducer
from L1Trigger.L1THGCalUtilities.hgcalTriggerNtuples_cfi import *

ntuple_multiclusters_fulltruth = ntuple_multiclusters.clone(
    Multiclusters = cms.InputTag('l1tCaloTruthCellsProducer'),
    Prefix = cms.untracked.string('cl3dfulltruth')
)
l1tHGCalTriggerNtuplizer.Ntuples.append(ntuple_multiclusters_fulltruth)

# If caloTruthCellsProducer.makeCellsCollection is True, can run the clustering algorithm over output cells too

if l1tCaloTruthCellsProducer.makeCellsCollection:
    ## ntuplize clusters and towers

    ntuple_triggercells.caloParticlesToCells = cms.InputTag('l1tCaloTruthCellsProducer')
    ntuple_triggercells.FillTruthMap = cms.bool(True)
    
    ntuple_triggercells_truth = ntuple_triggercells.clone(
        TriggerCells = cms.InputTag('l1tCaloTruthCellsProducer'),
        Multiclusters = cms.InputTag('l1tHGCalTruthBackEndLayer2Producer:HGCalBackendLayer2Processor3DClustering'),
        Prefix = cms.untracked.string('tctruth'),
        FillTruthMap = cms.bool(False)
    )
    
    ntuple_clusters_truth = ntuple_clusters.clone(
        Clusters = cms.InputTag('l1tHGCalTruthBackEndLayer1Producer:HGCalBackendLayer1Processor2DClustering'),
        Prefix = cms.untracked.string('cltruth')
    )
    
    ntuple_multiclusters_truth = ntuple_multiclusters.clone(
        Multiclusters = cms.InputTag('l1tHGCalTruthBackEndLayer2Producer:HGCalBackendLayer2Processor3DClustering'),
        Prefix = cms.untracked.string('cl3dtruth')
    )
    
    ntuple_towers_truth = ntuple_towers.clone(
        Towers = cms.InputTag('l1tHGCalTruthTowerProducer:HGCalTowerProcessor'),
        Prefix = cms.untracked.string('towertruth')
    )

    l1tHGCalTriggerNtuplizer.Ntuples.append(ntuple_triggercells_truth)
    l1tHGCalTriggerNtuplizer.Ntuples.append(ntuple_multiclusters_truth)
    l1tHGCalTriggerNtuplizer.Ntuples.append(ntuple_towers_truth)
