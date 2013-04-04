

full_title = "RecoParticleFlow collections (in RECO and AOD)"

full = {
    '0':['particleFlowClusterECAL', 'recoPFRecHits', 'No documentation'] ,
    '1':['particleFlowClusterHCAL', 'recoPFRecHits', 'No documentation'] ,
    '2':['particleFlowClusterHO', 'recoPFRecHits', 'No documentation'] ,
    '3':['particleFlowClusterHFEM', 'recoPFRecHits', 'No documentation'] ,
    '4':['particleFlowClusterHFHAD', 'recoPFRecHits', 'No documentation'] ,
    '5':['particleFlowClusterPS', 'recoPFRecHits', 'No documentation'] ,
    '6':['particleFlowRecHitECAL', 'recoPFRecHits', 'No documentation'] ,
    '7':['particleFlowRecHitHO', 'recoPFRecHits', 'No documentation'] ,
    '8':['particleFlowRecHitHCAL', 'recoPFRecHits', 'No documentation'] ,
    '9':['particleFlowRecHitPS', 'recoPFRecHits', 'No documentation'] ,
    '10':['particleFlowClusterECAL', 'recoPFClusters', 'No documentation'] ,
    '11':['particleFlowClusterHCAL', 'recoPFClusters', 'No documentation'] ,
    '12':['particleFlowClusterHO', 'recoPFClusters', 'No documentation'] ,
    '13':['particleFlowClusterHFEM', 'recoPFClusters', 'No documentation'] ,
    '14':['particleFlowClusterHFHAD', 'recoPFClusters', 'No documentation'] ,
    '15':['particleFlowClusterPS', 'recoPFClusters', 'No documentation'] ,
    '16':['particleFlowBlock', 'recoPFBlocks', 'No documentation'] ,
    '17':['particleFlow', 'recoPFCandidates', 'Particle Flow Candidates (refers to reco::PFBlock`s)'] ,
    '18':['particleFlowTmp', 'recoPFCandidates', 'No documentation'] ,
    '19':['particleFlowDisplacedVertex', 'recoPFDisplacedVertexs', 'No documentation'] ,
    
    '20':['pfElectronTranslator:pf','reco::SuperClusterCollection','Particle Flow Super-clusters of PF-electrons'],
    '21':['pfElectronTranslator:pf','reco::CaloClusterCollection','Basic clusters of PF electron super-clusters'],
    '22':['pfElectronTranslator:pf','reco::PreshowerClusterCollection','Preshower clusters of PF electron super-clusters'],
    
    '23':['pfPhotonTranslator:pfphot','reco::SuperClusterCollection','Particle Flow Super-clusters of photons imported in PF'],
    '24':['pfPhotonTranslator:pfphot','reco::CaloClusterCollection','Basic clusters of PF photon super-clusters'],
    '25':['pfPhotonTranslator:pfphot','reco::PreshowerClusterCollection','Preshower clusters of PF photon super-clusters'],
    '26':['pfPhotonTranslator:pfphot','reco::PhotonCollection','photons imported in PF translated into the RECO format'],
    
    '27':['particleFlow:electrons','reco::PFCandidateCollection','Particle Flow Electron Candidates without any selection (refers to reco::PFBlock`s)'],
    '28':['particleFlow', '*', 'No documentation'] ,
    '29':['trackerDrivenElectronSeeds', '*', 'No documentation'] ,
    '30':['particleFlowPtrs', '*', 'No documentation'] ,
    '31':['particleFlowTmpPtrs', '*', 'No documentation'] 
}

reco_title = "RecoParticleFlow collections (in RECO only)"

reco = {
    '0':['particleFlowClusterECAL', 'recoPFRecHits', 'No documentation'] ,
    '1':['particleFlowClusterHCAL', 'recoPFRecHits', 'No documentation'] ,
    '2':['particleFlowClusterHO', 'recoPFRecHits', 'No documentation'] ,
    '3':['particleFlowClusterHFEM', 'recoPFRecHits', 'No documentation'] ,
    '4':['particleFlowClusterHFHAD', 'recoPFRecHits', 'No documentation'] ,
    '5':['particleFlowClusterPS', 'recoPFRecHits', 'No documentation'] ,
    '6':['particleFlowRecHitECAL', 'recoPFRecHits', 'No documentation'] ,
    '7':['particleFlowRecHitHCAL', 'recoPFRecHits', 'No documentation'] ,
    '8':['particleFlowRecHitHO', 'recoPFRecHits', 'No documentation'] ,
    '9':['particleFlowRecHitPS', 'recoPFRecHits', 'No documentation'] ,
    '10':['particleFlowClusterECAL', 'recoPFClusters', 'ECAL clusters'] ,
    '11':['particleFlowClusterHCAL', 'recoPFClusters', 'HCAL clusters'] ,
    '12':['particleFlowClusterHO', 'recoPFClusters', 'No documentation'] ,
    '13':['particleFlowClusterPS', 'recoPFClusters', 'Preshower clusters'] ,
    '14':['particleFlowBlock', 'recoPFBlocks', 'Particle Flow Blocks (refers to reco::Track`s and reco::PFCluster`s)'] ,
    '15':['particleFlow', 'recoPFCandidates', 'Particle Flow Candidates (refers to reco::PFBlock`s)'] ,
    '16':['particleFlowTmp', 'recoPFCandidates', 'No documentation'] ,
    '17':['particleFlowTmp', 'recoPFCandidates', 'No documentation'] ,
    '18':['particleFlowDisplacedVertex', 'recoPFDisplacedVertexs', 'No documentation'] ,
    '19':['pfElectronTranslator', 'reco::GsfElectronCollection','PF electron reconstructed translated in the GsfElectron format'],
    '20':['pfPhotonTranslator', '*', 'No documentation'] ,
    '21':['particleFlow', '*', 'No documentation'] ,
    '22':['particleFlow', '*', 'No documentation'] ,
    '23':['trackerDrivenElectronSeeds', 'reco::PreIdCollection','Information on the tracker-driven electron seed'],
    '24':['particleFlowPtrs', '*', 'No documentation'] ,
    '25':['particleFlowTmpPtrs', '*', 'No documentation'] 
}

aod_title = "RecoParticleFlow collections (in AOD only)"

aod = {
    '0':['particleFlowClusterECAL', 'recoPFRecHits', 'No documentation'] ,
    '1':['particleFlowClusterHCAL', 'recoPFRecHits', 'No documentation'] ,
    '2':['particleFlowClusterHO', 'recoPFRecHits', 'No documentation'] ,
    '3':['particleFlowClusterHFEM', 'recoPFRecHits', 'No documentation'] ,
    '4':['particleFlowClusterHFHAD', 'recoPFRecHits', 'No documentation'] ,
    '5':['particleFlowClusterPS', 'recoPFRecHits', 'No documentation'] ,
    '6':['particleFlowRecHitECAL', 'recoPFRecHits', 'No documentation'] ,
    '7':['particleFlowRecHitHCAL', 'recoPFRecHits', 'No documentation'] ,
    '8':['particleFlowRecHitHO', 'recoPFRecHits', 'No documentation'] ,
    '9':['particleFlowRecHitPS', 'recoPFRecHits', 'No documentation'] ,
    '10':['particleFlow', 'recoPFCandidates', 'No documentation'] ,
    '11':['particleFlowTmp', 'recoPFCandidates', 'No documentation'] ,
    '12':['particleFlow', '*', 'No documentation'] ,
    '13':['particleFlow', '*', 'No documentation'] ,
    '14':['pfElectronTranslator','reco::SuperClusterCollection','Particle Flow Super-clusters of PF-electrons'],
    '15':['pfElectronTranslator','reco::CaloClusterCollection','Basic clusters of PF electron super-clusters'],
    '16':['pfElectronTranslator','reco::PreshowerClusterCollection','Preshower clusters of PF electron super-clusters'],
    '17':['pfPhotonTranslator:pfphot','reco::CaloClusterCollection','Basic clusters of PF photon super-clusters'],
    '18':['pfPhotonTranslator:pfphot','reco::PreshowerClusterCollection','Preshower clusters of PF photon super-clusters'],
    '19':['pfPhotonTranslator:pfphot','reco::SuperClusterCollection','Particle Flow Super-clusters of photons imported in PF'],
    '20':['pfPhotonTranslator:pfphot','reco::PhotonCollection','photons imported in PF translated into the RECO format'],
    '21':['pfPhotonTranslator', 'recoPhotonCores', 'No documentation'] ,
    '22':['pfPhotonTranslator', 'recoConversions', 'No documentation'] ,
    '23':['particleFlowPtrs', '*', 'No documentation'] ,
    '24':['particleFlowTmpPtrs', '*', 'No documentation'] 
}





