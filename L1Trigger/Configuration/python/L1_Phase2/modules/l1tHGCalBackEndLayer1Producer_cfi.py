import FWCore.ParameterSet.Config as cms

l1tHGCalBackEndLayer1Producer = cms.EDProducer("HGCalBackendLayer1Producer",
    InputTriggerCells = cms.InputTag("l1tHGCalConcentratorProducer","HGCalConcentratorProcessorSelection"),
    ProcessorParameters = cms.PSet(
        C2d_parameters = cms.PSet(
            applyLayerCalibration = cms.bool(False),
            calibSF_cluster = cms.double(1.0),
            clusterType = cms.string('dummyC2d'),
            layerWeights = cms.vdouble(
                0.0, 0.0183664, 0.0, 0.0305622, 0.0,
                0.0162589, 0.0, 0.0143918, 0.0, 0.0134475,
                0.0, 0.0185754, 0.0, 0.0204934, 0.0,
                0.016901, 0.0, 0.0207958, 0.0, 0.0167985,
                0.0, 0.0238385, 0.0, 0.0301146, 0.0,
                0.0274622, 0.0, 0.0468671, 0.0, 0.078819,
                0.0555831, 0.0609312, 0.0610768, 0.0657626, 0.0465653,
                0.0629383, 0.0610061, 0.0517326, 0.0492882, 0.0699336,
                0.0673457, 0.119896, 0.125327, 0.143235, 0.153295,
                0.104777, 0.109345, 0.161386, 0.174656, 0.108053,
                0.121674, 0.1171, 0.328053
            )
        ),
        ProcessorName = cms.string('HGCalBackendLayer1Processor2DClustering')
    )
)
