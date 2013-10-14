'''
    Created on Jun 26, 2013 
    @author:  Mantas Stankevicius
    @contact: mantas.stankevicius@cern.ch
    http://cmsdoxy.web.cern.ch/cmsdoxy/dataformats/
    
    @responsible: 
    
'''

json = {
  "full": {
    "title": "RecoParticleFlow collections (in RECO and AOD)",
    "data": [
     {
      "instance": "pfPhotonTranslator:pfphot",
      "container": "reco::CaloClusterCollection",
      "desc": "Basic clusters of PF photon super-clusters"
     },
     {
      "instance": "particleFlowPtrs",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "pfPhotonTranslator:pfphot",
      "container": "reco::PreshowerClusterCollection",
      "desc": "Preshower clusters of PF photon super-clusters"
     },
     {
      "instance": "pfElectronTranslator:pf",
      "container": "reco::PreshowerClusterCollection",
      "desc": "Preshower clusters of PF electron super-clusters"
     },
     {
      "instance": "pfPhotonTranslator:pfphot",
      "container": "reco::PhotonCollection",
      "desc": "photons imported in PF translated into the RECO format"
     },
     {
      "instance": "particleFlow:electrons",
      "container": "reco::PFCandidateCollection",
      "desc": "Particle Flow Electron Candidates without any selection (refers to reco::PFBlock`s)"
     },
     {
      "instance": "pfElectronTranslator:pf",
      "container": "reco::SuperClusterCollection",
      "desc": "Particle Flow Super-clusters of PF-electrons"
     },
     {
      "instance": "pfElectronTranslator:pf",
      "container": "reco::CaloClusterCollection",
      "desc": "Basic clusters of PF electron super-clusters"
     },
     {
      "instance": "particleFlowClusterHCAL",
      "container": "recoPFClusters",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterECAL",
      "container": "recoPFClusters",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterHFEM",
      "container": "recoPFClusters",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterHO",
      "container": "recoPFClusters",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterPS",
      "container": "recoPFClusters",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterHFHAD",
      "container": "recoPFClusters",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlow",
      "container": "recoPFCandidates",
      "desc": "Particle Flow Candidates (refers to reco::PFBlock`s)"
     },
     {
      "instance": "particleFlowBlock",
      "container": "recoPFBlocks",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowDisplacedVertex",
      "container": "recoPFDisplacedVertexs",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowTmp",
      "container": "recoPFCandidates",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowTmpPtrs",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "pfPhotonTranslator:pfphot",
      "container": "reco::SuperClusterCollection",
      "desc": "Particle Flow Super-clusters of photons imported in PF"
     },
     {
      "instance": "particleFlow",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "trackerDrivenElectronSeeds",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterHCAL",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterECAL",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterHFEM",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterHO",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterPS",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterHFHAD",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowRecHitHO",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowRecHitECAL",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowRecHitPS",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowRecHitHCAL",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     }
    ]
  },
  "aod": {
    "title": "RecoParticleFlow collections (in AOD only)",
    "data": [
     {
      "instance": "particleFlowTmpPtrs",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "pfPhotonTranslator:pfphot",
      "container": "reco::PhotonCollection",
      "desc": "photons imported in PF translated into the RECO format"
     },
     {
      "instance": "pfPhotonTranslator",
      "container": "recoPhotonCores",
      "desc": "No documentation"
     },
     {
      "instance": "pfPhotonTranslator",
      "container": "recoConversions",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowPtrs",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterHCAL",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterECAL",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterHFEM",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterHO",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterPS",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterHFHAD",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowRecHitHCAL",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowRecHitECAL",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowRecHitPS",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowRecHitHO",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowTmp",
      "container": "recoPFCandidates",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlow",
      "container": "recoPFCandidates",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlow",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlow",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "pfElectronTranslator",
      "container": "reco::CaloClusterCollection",
      "desc": "Basic clusters of PF electron super-clusters"
     },
     {
      "instance": "pfElectronTranslator",
      "container": "reco::SuperClusterCollection",
      "desc": "Particle Flow Super-clusters of PF-electrons"
     },
     {
      "instance": "pfPhotonTranslator:pfphot",
      "container": "reco::CaloClusterCollection",
      "desc": "Basic clusters of PF photon super-clusters"
     },
     {
      "instance": "pfElectronTranslator",
      "container": "reco::PreshowerClusterCollection",
      "desc": "Preshower clusters of PF electron super-clusters"
     },
     {
      "instance": "pfPhotonTranslator:pfphot",
      "container": "reco::SuperClusterCollection",
      "desc": "Particle Flow Super-clusters of photons imported in PF"
     },
     {
      "instance": "pfPhotonTranslator:pfphot",
      "container": "reco::PreshowerClusterCollection",
      "desc": "Preshower clusters of PF photon super-clusters"
     }
    ]
  },
  "reco": {
    "title": "RecoParticleFlow collections (in RECO only)",
    "data": [
     {
      "instance": "particleFlowPtrs",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowTmpPtrs",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "pfPhotonTranslator",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlow",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlow",
      "container": "*",
      "desc": "No documentation"
     },
     {
      "instance": "trackerDrivenElectronSeeds",
      "container": "reco::PreIdCollection",
      "desc": "Information on the tracker-driven electron seed"
     },
     {
      "instance": "particleFlowClusterHCAL",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterECAL",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterHFEM",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterHO",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterPS",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterHFHAD",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowRecHitHCAL",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowRecHitECAL",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowRecHitPS",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowRecHitHO",
      "container": "recoPFRecHits",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowClusterHCAL",
      "container": "recoPFClusters",
      "desc": "HCAL clusters"
     },
     {
      "instance": "particleFlowClusterECAL",
      "container": "recoPFClusters",
      "desc": "ECAL clusters"
     },
     {
      "instance": "particleFlowClusterPS",
      "container": "recoPFClusters",
      "desc": "Preshower clusters"
     },
     {
      "instance": "particleFlowClusterHO",
      "container": "recoPFClusters",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlow",
      "container": "recoPFCandidates",
      "desc": "Particle Flow Candidates (refers to reco::PFBlock`s)"
     },
     {
      "instance": "particleFlowBlock",
      "container": "recoPFBlocks",
      "desc": "Particle Flow Blocks (refers to reco::Track`s and reco::PFCluster`s)"
     },
     {
      "instance": "particleFlowTmp",
      "container": "recoPFCandidates",
      "desc": "No documentation"
     },
     {
      "instance": "particleFlowTmp",
      "container": "recoPFCandidates",
      "desc": "No documentation"
     },
     {
      "instance": "pfElectronTranslator",
      "container": "reco::GsfElectronCollection",
      "desc": "PF electron reconstructed translated in the GsfElectron format"
     },
     {
      "instance": "particleFlowDisplacedVertex",
      "container": "recoPFDisplacedVertexs",
      "desc": "No documentation"
     }
    ]
  }
}
