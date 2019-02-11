import FWCore.ParameterSet.Config as cms

binSums = cms.vuint32(13,               #0
                      11, 11, 11,       # 1 - 3
                      9, 9, 9,          # 4 - 6
                      7, 7, 7, 7, 7, 7, # 7 - 12
                      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,  # 13 - 27
                      3, 3, 3, 3, 3, 3, 3, 3  # 28 - 35
                      )


def custom_2dclustering_distance(process, 
        distance=6.,# cm
        seed_threshold=5.,# MipT
        cluster_threshold=2.# MipT
        ):
    parameters_c2d = process.hgcalBackEndLayer1Producer.ProcessorParameters.C2d_parameters
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
    parameters_c2d = process.hgcalBackEndLayer1Producer.ProcessorParameters.C2d_parameters
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
    parameters_c2d = process.hgcalBackEndLayer1Producer.ProcessorParameters.C2d_parameters
    parameters_c2d.seeding_threshold_silicon = cms.double(seed_threshold) # MipT
    parameters_c2d.seeding_threshold_scintillator = cms.double(seed_threshold) # MipT
    parameters_c2d.clustering_threshold_silicon = cms.double(cluster_threshold) # MipT
    parameters_c2d.clustering_threshold_scintillator = cms.double(cluster_threshold) # MipT
    parameters_c2d.dR_cluster = cms.double(distance) # cm
    parameters_c2d.clusterType = cms.string('dRNNC2d') 
    return process

def custom_2dclustering_dummy(process):
    parameters_c2d = process.hgcalBackEndLayer1Producer.ProcessorParameters.C2d_parameters
    parameters_c2d.clusterType = cms.string('dummyC2d')
    return process


def custom_3dclustering_distance(process,
        distance=0.01
        ):
    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    parameters_c3d.dR_multicluster = cms.double(distance)
    parameters_c3d.type_multicluster = cms.string('dRC3d')
    return process


def custom_3dclustering_dbscan(process,
        distance=0.005,
        min_points=3
        ):
    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    parameters_c3d.dist_dbscan_multicluster = cms.double(distance)
    parameters_c3d.minN_dbscan_multicluster = cms.uint32(min_points)
    parameters_c3d.type_multicluster = cms.string('DBSCANC3d')
    return process


def custom_3dclustering_histoMax(process,
        distance = 0.01,
        nBins_R = 36,
        nBins_Phi = 216,
        binSumsHisto = binSums,                        
        seed_threshold = 0,
        ):
    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    parameters_c3d.dR_multicluster = cms.double(distance)
    parameters_c3d.nBins_R_histo_multicluster = cms.uint32(nBins_R)
    parameters_c3d.nBins_Phi_histo_multicluster = cms.uint32(nBins_Phi)
    parameters_c3d.binSumsHisto = binSumsHisto
    parameters_c3d.threshold_histo_multicluster = seed_threshold
    parameters_c3d.type_multicluster = cms.string('HistoMaxC3d')
    return process

def custom_3dclustering_histoMax_variableDr(process,
        distances = ([0] + [0.010]*7 + [0.020]*7 + [0.030]*7 + [0.040]*7 +   [0.040]*6 + [0.050]*6  +  [0.050]*12),
        nBins_R = 36,
        nBins_Phi = 216,
        binSumsHisto = binSums,                        
        seed_threshold = 0,
        ):
    process = custom_3dclustering_histoMax(process, 0, nBins_R, nBins_Phi, binSumsHisto, seed_threshold)
    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    parameters_c3d.dR_multicluster_byLayer = cms.vdouble(distances)
    return process


def custom_3dclustering_histoInterpolatedMax(process,
        distance = 0.01,
        nBins_R = 36,
        nBins_Phi = 216,
        binSumsHisto = binSums,
        ):
    process = custom_3dclustering_histoMax( process, distance, nBins_R, nBins_Phi, binSumsHisto )    
    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    parameters_c3d.type_multicluster = cms.string('HistoInterpolatedMaxC3d')
    return process

def custom_3dclustering_histoInterpolatedMax1stOrder(process):

    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    parameters_c3d.neighbour_weights=cms.vdouble(  0    , 0.25, 0   ,
                                                   0.25 , 0   , 0.25,
                                                   0    , 0.25, 0
                                                )
    process = custom_3dclustering_histoInterpolatedMax( process )    
    return process



def custom_3dclustering_histoInterpolatedMax2ndOrder(process):

    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    parameters_c3d.neighbour_weights=cms.vdouble( -0.25, 0.5, -0.25,
                                                   0.5 , 0  ,  0.5 ,
                                                  -0.25, 0.5, -0.25
                                                )
    process = custom_3dclustering_histoInterpolatedMax( process )    
    return process



def custom_3dclustering_histoThreshold(process,
        threshold = 20.,
        distance = 0.01,
        nBins_R = 36,
        nBins_Phi = 216,
        binSumsHisto = binSums,
        ):
    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    parameters_c3d.threshold_histo_multicluster = cms.double(threshold)
    parameters_c3d.dR_multicluster = cms.double(distance)
    parameters_c3d.nBins_R_histo_multicluster = cms.uint32(nBins_R)
    parameters_c3d.nBins_Phi_histo_multicluster = cms.uint32(nBins_Phi)
    parameters_c3d.binSumsHisto = binSumsHisto
    parameters_c3d.type_multicluster = cms.string('HistoThresholdC3d')
    return process
