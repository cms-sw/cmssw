#include "RecoEgamma/EgammaTools/interface/BaselinePFSCRegression.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "TVector2.h"
#include "DataFormats/Math/interface/deltaR.h"

void BaselinePFSCRegression::update(const edm::EventSetup& es) {
  const CaloTopologyRecord& topofrom_es = es.get<CaloTopologyRecord>();
  if( !topo_record ||
      topofrom_es.cacheIdentifier() != topo_record->cacheIdentifier() ) {
    topo_record = &topofrom_es;
    topo_record->get(calotopo);
  }
  const CaloGeometryRecord& geomfrom_es = es.get<CaloGeometryRecord>();
  if( !geom_record ||
      geomfrom_es.cacheIdentifier() != geom_record->cacheIdentifier() ) {
    geom_record = &geomfrom_es;
    geom_record->get(calogeom);
  }
}

void BaselinePFSCRegression::set(const reco::SuperCluster& sc,
				 std::vector<float>& vars     ) const {
  vars.clear();
  vars.resize(33);
  const double rawEnergy = sc.rawEnergy(), calibEnergy = sc.correctedEnergy();
  const edm::Ptr<reco::CaloCluster> &seed = sc.seed();
  const size_t nVtx = vertices->size();
  float maxDR=999., maxDRDPhi=999., maxDRDEta=999., maxDRRawEnergy=0.;
  float subClusRawE[3], subClusDPhi[3], subClusDEta[3];
  memset(subClusRawE,0,3*sizeof(float));
  memset(subClusDPhi,0,3*sizeof(float));
  memset(subClusDEta,0,3*sizeof(float));
  size_t iclus=0;
  for( auto clus = sc.clustersBegin()+1; clus != sc.clustersEnd(); ++clus ) {
    const float this_deta = (*clus)->eta() - seed->eta();
    const float this_dphi = TVector2::Phi_mpi_pi((*clus)->phi() - seed->phi());
    const float this_dr = std::hypot(this_deta,this_dphi);
    if(this_dr > maxDR || maxDR == 999.0f) {
      maxDR = this_dr;
      maxDRDEta = this_deta;
      maxDRDPhi = this_dphi;
      maxDRRawEnergy = (*clus)->energy();
    }
    if( iclus++ < 3 ) {
      subClusRawE[iclus] = (*clus)->energy();
      subClusDEta[iclus] = this_deta;
      subClusDPhi[iclus] = this_dphi;
    }
  }
  float scPreshowerSum = sc.preshowerEnergy();
  switch( seed->hitsAndFractions().at(0).first.subdetId() ) {
  case EcalBarrel:
    {
      const float eMax = EcalClusterTools::eMax( *seed, &*rechitsEB );
      const float e2nd = EcalClusterTools::e2nd( *seed, &*rechitsEB );
      const float e3x3 = EcalClusterTools::e3x3( *seed,
						 &*rechitsEB, 
						 &*calotopo  );
      const float eTop = EcalClusterTools::eTop( *seed, 
						 &*rechitsEB, 
						 &*calotopo );
      const float eBottom = EcalClusterTools::eBottom( *seed, 
						       &*rechitsEB, 
						       &*calotopo );
      const float eLeft = EcalClusterTools::eLeft( *seed, 
						   &*rechitsEB, 
						   &*calotopo );
      const float eRight = EcalClusterTools::eRight( *seed, 
						     &*rechitsEB, 
						     &*calotopo );
      const float eLpeR = eLeft + eRight;
      const float eTpeB = eTop + eBottom;
      const float eLmeR = eLeft - eRight;
      const float eTmeB = eTop - eBottom;
      std::vector<float> vCov = 
	EcalClusterTools::localCovariances( *seed, &*rechitsEB, &*calotopo );
      const float see = (isnan(vCov[0]) ? 0. : sqrt(vCov[0]));
      const float spp = (isnan(vCov[2]) ? 0. : sqrt(vCov[2]));
      float sep = 0.;
      if (see*spp > 0)
        sep = vCov[1] / (see * spp);
      else if (vCov[1] > 0)
        sep = 1.0;
      else
        sep = -1.0;
      float cryPhi, cryEta, thetatilt, phitilt;
      int ieta, iphi;
      ecl_.localCoordsEB(*seed, *calogeom, cryEta, cryPhi, 
			 ieta, iphi, thetatilt, phitilt);
      vars[0] = nVtx;                          //nVtx
      vars[1] = sc.eta();                      //scEta
      vars[2] = sc.phi();                      //scPhi
      vars[3] = sc.etaWidth();                 //scEtaWidth
      vars[4] = sc.phiWidth();                 //scPhiWidth
      vars[5] = e3x3/rawEnergy;                //scSeedR9
      vars[6] = sc.seed()->energy()/rawEnergy; //scSeedRawEnergy/scRawEnergy
      vars[7] = eMax/rawEnergy;                //scSeedEmax/scRawEnergy
      vars[8] = e2nd/rawEnergy;                //scSeedE2nd/scRawEnergy
      vars[9] = (eLpeR!=0. ? eLmeR/eLpeR : 0.);//scSeedLeftRightAsym
      vars[10] = (eTpeB!=0.? eTmeB/eTpeB : 0.);//scSeedTopBottomAsym
      vars[11] = see;                          //scSeedSigmaIetaIeta
      vars[12] = sep;                          //scSeedSigmaIetaIphi
      vars[13] = spp;                          //scSeedSigmaIphiIphi
      vars[14] = sc.clustersSize()-1;          //N_ECALClusters
      vars[15] = maxDR;                        //clusterMaxDR
      vars[16] = maxDRDPhi;                    //clusterMaxDRDPhi
      vars[17] = maxDRDEta;                    //clusterMaxDRDEta
      vars[18] = maxDRRawEnergy/rawEnergy; //clusMaxDRRawEnergy/scRawEnergy
      vars[19] = subClusRawE[0]/rawEnergy; //clusterRawEnergy[0]/scRawEnergy
      vars[20] = subClusRawE[1]/rawEnergy; //clusterRawEnergy[1]/scRawEnergy
      vars[21] = subClusRawE[2]/rawEnergy; //clusterRawEnergy[2]/scRawEnergy
      vars[22] = subClusDPhi[0];               //clusterDPhiToSeed[0]
      vars[23] = subClusDPhi[1];               //clusterDPhiToSeed[1]
      vars[24] = subClusDPhi[2];               //clusterDPhiToSeed[2]
      vars[25] = subClusDEta[0];               //clusterDEtaToSeed[0]
      vars[26] = subClusDEta[1];               //clusterDEtaToSeed[1]
      vars[27] = subClusDEta[2];               //clusterDEtaToSeed[2]
      vars[28] = cryEta;                       //scSeedCryEta
      vars[29] = cryPhi;                       //scSeedCryPhi
      vars[30] = ieta;                         //scSeedCryIeta
      vars[31] = iphi;                         //scSeedCryIphi
      vars[32] = calibEnergy;                  //scCalibratedEnergy
    }
    break;
  case EcalEndcap:
    {
      const float eMax = EcalClusterTools::eMax( *seed, &*rechitsEE );
      const float e2nd = EcalClusterTools::e2nd( *seed, &*rechitsEE );
      const float e3x3 = EcalClusterTools::e3x3( *seed,
						 &*rechitsEE, 
						 &*calotopo  );
      const float eTop = EcalClusterTools::eTop( *seed, 
						 &*rechitsEE, 
						 &*calotopo );
      const float eBottom = EcalClusterTools::eBottom( *seed, 
						       &*rechitsEE, 
						       &*calotopo );
      const float eLeft = EcalClusterTools::eLeft( *seed, 
						   &*rechitsEE, 
						   &*calotopo );
      const float eRight = EcalClusterTools::eRight( *seed, 
						     &*rechitsEE, 
						     &*calotopo );
      const float eLpeR = eLeft + eRight;
      const float eTpeB = eTop + eBottom;
      const float eLmeR = eLeft - eRight;
      const float eTmeB = eTop - eBottom;
      std::vector<float> vCov = 
	EcalClusterTools::localCovariances( *seed, &*rechitsEE, &*calotopo );
      const float see = (isnan(vCov[0]) ? 0. : sqrt(vCov[0]));
      const float spp = (isnan(vCov[2]) ? 0. : sqrt(vCov[2]));
      float sep = 0.;
      if (see*spp > 0)
        sep = vCov[1] / (see * spp);
      else if (vCov[1] > 0)
        sep = 1.0;
      else
        sep = -1.0;      
      vars[0] = nVtx;                          //nVtx
      vars[1] = sc.eta();                      //scEta
      vars[2] = sc.phi();                      //scPhi
      vars[3] = sc.etaWidth();                 //scEtaWidth
      vars[4] = sc.phiWidth();                 //scPhiWidth
      vars[5] = e3x3/rawEnergy;                //scSeedR9
      vars[6] = sc.seed()->energy()/rawEnergy; //scSeedRawEnergy/scRawEnergy
      vars[7] = eMax/rawEnergy;                //scSeedEmax/scRawEnergy
      vars[8] = e2nd/rawEnergy;                //scSeedE2nd/scRawEnergy
      vars[9] = (eLpeR!=0. ? eLmeR/eLpeR : 0.);//scSeedLeftRightAsym
      vars[10] = (eTpeB!=0.? eTmeB/eTpeB : 0.);//scSeedTopBottomAsym
      vars[11] = see;                          //scSeedSigmaIetaIeta
      vars[12] = sep;                          //scSeedSigmaIetaIphi
      vars[13] = spp;                          //scSeedSigmaIphiIphi
      vars[14] = sc.clustersSize()-1;          //N_ECALClusters
      vars[15] = maxDR;                        //clusterMaxDR
      vars[16] = maxDRDPhi;                    //clusterMaxDRDPhi
      vars[17] = maxDRDEta;                    //clusterMaxDRDEta
      vars[18] = maxDRRawEnergy/rawEnergy; //clusMaxDRRawEnergy/scRawEnergy
      vars[19] = subClusRawE[0]/rawEnergy; //clusterRawEnergy[0]/scRawEnergy
      vars[20] = subClusRawE[1]/rawEnergy; //clusterRawEnergy[1]/scRawEnergy
      vars[21] = subClusRawE[2]/rawEnergy; //clusterRawEnergy[2]/scRawEnergy
      vars[22] = subClusDPhi[0];               //clusterDPhiToSeed[0]
      vars[23] = subClusDPhi[1];               //clusterDPhiToSeed[1]
      vars[24] = subClusDPhi[2];               //clusterDPhiToSeed[2]
      vars[25] = subClusDEta[0];               //clusterDEtaToSeed[0]
      vars[26] = subClusDEta[1];               //clusterDEtaToSeed[1]
      vars[27] = subClusDEta[2];               //clusterDEtaToSeed[2]
      vars[28] = scPreshowerSum/rawEnergy;   //scPreshowerEnergy/scRawEnergy
      vars[29] = calibEnergy;                  //scCalibratedEnergy
    }    
    break;    
  default:
   throw cms::Exception("PFECALSuperClusterProducer::calculateRegressedEnergy")
     << "Supercluster seed is either EB nor EE!" << std::endl;
  }
}

void BaselinePFSCRegression::
setTokens(const edm::ParameterSet& ps, edm::ConsumesCollector&& cc) {
  const edm::InputTag rceb = ps.getParameter<edm::InputTag>("ecalRecHitsEB");
  const edm::InputTag rcee = ps.getParameter<edm::InputTag>("ecalRecHitsEE");
  const edm::InputTag vtx = ps.getParameter<edm::InputTag>("vertexCollection");
  inputTagEBRecHits_      = cc.consumes<EcalRecHitCollection>(rceb);
  inputTagEERecHits_      = cc.consumes<EcalRecHitCollection>(rcee);
  inputTagVertices_       = cc.consumes<reco::VertexCollection>(vtx);
}

void BaselinePFSCRegression::setEvent(const edm::Event& ev) {
  ev.getByToken(inputTagEBRecHits_,rechitsEB);
  ev.getByToken(inputTagEERecHits_,rechitsEE);
  ev.getByToken(inputTagVertices_,vertices);
}
