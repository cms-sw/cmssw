import FWCore.ParameterSet.Config as cms

def create_genmatch(process, inputs,
        distance=0.3
        ):
    producer = process.hgc3DClusterGenMatchSelector.clone() 
    producer.dR = cms.double(distance)
    producer.src = cms.InputTag('{}:HGCalBackendLayer2Processor3DClustering'.format(inputs))
    return producer
