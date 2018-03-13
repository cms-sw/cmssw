import FWCore.ParameterSet.Config as cms


def custom_2dclustering_distance(process, 
        distance=6.,# cm
        seed_threshold=5.,# MipT
        cluster_threshold=2.# MipT
        ):
    parameters_c2d = process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C2d_parameters
    parameters_c2d.seeding_threshold_silicon = cms.double(seed_threshold) 
    parameters_c2d.seeding_threshold_scintillator = cms.double(seed_threshold) 
    parameters_c2d.clustering_threshold_silicon = cms.double(cluster_threshold) 
    parameters_c2d.clustering_threshold_scintillator = cms.double(cluster_threshold) 
    parameters_c2d.dR_cluster = cms.double(distance) 
    parameters_c2d.clusterType = cms.string('dRC2d') 
    return process

def custom_2dclustering_topological(process,
        seed_threshold=5.,# MipT
        cluster_threshold=2.# MipT
        ):
    parameters_c2d = process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C2d_parameters
    parameters_c2d.seeding_threshold_silicon = cms.double(seed_threshold) # MipT
    parameters_c2d.seeding_threshold_scintillator = cms.double(seed_threshold) # MipT
    parameters_c2d.clustering_threshold_silicon = cms.double(cluster_threshold) # MipT
    parameters_c2d.clustering_threshold_scintillator = cms.double(cluster_threshold) # MipT
    parameters_c2d.clusterType = cms.string('NNC2d') 
    return process

def custom_2dclustering_constrainedtopological(process,
        distance=6.,# cm
        seed_threshold=5.,# MipT
        cluster_threshold=2.# MipT
        ):
    parameters_c2d = process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C2d_parameters
    parameters_c2d.seeding_threshold_silicon = cms.double(seed_threshold) # MipT
    parameters_c2d.seeding_threshold_scintillator = cms.double(seed_threshold) # MipT
    parameters_c2d.clustering_threshold_silicon = cms.double(cluster_threshold) # MipT
    parameters_c2d.clustering_threshold_scintillator = cms.double(cluster_threshold) # MipT
    parameters_c2d.dR_cluster = cms.double(distance) # cm
    parameters_c2d.clusterType = cms.string('dRNNC2d') 
    return process

def custom_3dclustering_distance(process,
        distance=0.01
        ):
    parameters_c3d = process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C3d_parameters
    parameters_c3d.dR_multicluster = cms.double(distance)
    parameters_c3d.type_multicluster = cms.string('dRC3d')
    return process


def custom_3dclustering_dbscan(process,
        distance=0.005,
        min_points=3
        ):
    parameters_c3d = process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C3d_parameters
    parameters_c3d.dist_dbscan_multicluster = cms.double(distance)
    parameters_c3d.minN_dbscan_multicluster = cms.uint32(min_points)
    parameters_c3d.type_multicluster = cms.string('DBSCANC3d')
    return process
