import FWCore.ParameterSet.Config as cms
from L1Trigger.L1THGCalUtilities.caloTruthCellsProducer_cfi import caloTruthCellsProducer

def create_truth_prod(process, inputs):
    producer = process.caloTruthCellsProducer.clone(
            triggerCells = cms.InputTag('{}:HGCalVFEProcessorSums'.format(inputs))
            )
    return producer
