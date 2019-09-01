import FWCore.ParameterSet.Config as cms

def create_genmatch(process, inputs,
        distance=0.3
        ):
    producer = process.hgc3DClusterGenMatchSelector.clone(
            dR = cms.double(distance),
            src = cms.InputTag('{}:HGCalBackendLayer2Processor3DClustering'.format(inputs))
            )
    return producer
