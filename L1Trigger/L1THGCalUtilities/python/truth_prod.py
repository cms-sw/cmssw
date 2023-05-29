import FWCore.ParameterSet.Config as cms
from L1Trigger.L1THGCalUtilities.caloTruthCellsProducer_cfi import l1tCaloTruthCellsProducer

def create_truth_prod(process, inputs):
    producer = process.l1tCaloTruthCellsProducer.clone(
            triggerCells = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
            )
    return producer
