#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/RegressionHelper.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "TVector2.h"
#include "TFile.h"

RegressionHelper::RegressionHelper( const Configuration & config):
  cfg_(config),ecalRegressionInitialized_(false),combinationRegressionInitialized_(false),
  caloTopologyCacheId_(0), caloGeometryCacheId_(0),regressionCacheId_(0)
{;}

RegressionHelper::~RegressionHelper(){;}

void RegressionHelper::applyEcalRegression(reco::GsfElectron & ele,
					   const edm::Handle<reco::VertexCollection>& vertices,
					   const edm::Handle<EcalRecHitCollection>& rechitsEB,
					   const edm::Handle<EcalRecHitCollection>& rechitsEE) const {
  double cor, err;
  getEcalRegression(*ele.superCluster(), vertices, rechitsEB, rechitsEE, cor, err);
  ele.setCorrectedEcalEnergy(cor * ele.superCluster()->correctedEnergy());
  ele.setCorrectedEcalEnergyError( err * ele.superCluster()->correctedEnergy());	

}

void RegressionHelper::checkSetup(const edm::EventSetup & es) {

  // Topology 
  unsigned long long newCaloTopologyCacheId 
    = es.get<CaloTopologyRecord>().cacheIdentifier() ;
  if (!caloTopologyCacheId_ || ( caloTopologyCacheId_ != newCaloTopologyCacheId ) )
    {
      caloTopologyCacheId_ = newCaloTopologyCacheId;
      edm::ESHandle<CaloTopology> caloTopo;
      es.get<CaloTopologyRecord>().get(caloTopo);
      caloTopology_ = &(*caloTopo);
    }
    
  // Geometry
  unsigned long long newCaloGeometryCacheId
    = es.get<CaloGeometryRecord>().cacheIdentifier() ;
  if (!caloGeometryCacheId_ || ( caloGeometryCacheId_ != newCaloGeometryCacheId ) ) 
    {
      caloGeometryCacheId_ = newCaloGeometryCacheId;
      edm::ESHandle<CaloGeometry> caloGeom;
      es.get<CaloGeometryRecord>().get(caloGeom);
      caloGeometry_ = &(*caloGeom);
    }

  // Ecal regression 
  
  // if at least one of the set of weights come from the DB
  if(cfg_.ecalWeightsFromDB)
    {
      unsigned long long newRegressionCacheId
	= es.get<GBRWrapperRcd>().cacheIdentifier() ;
      if ( !regressionCacheId_ || (newRegressionCacheId != regressionCacheId_ ) ) {
	
	const GBRWrapperRcd& gbrRcd = es.get<GBRWrapperRcd>();

	// ECAL barrel
	edm::ESHandle<GBRForest> ecalRegBarrelH;
	gbrRcd.get(cfg_.ecalRegressionWeightLabels[0].c_str(),ecalRegBarrelH);
	ecalRegBarrel_ = &(*ecalRegBarrelH);
	
	// ECAL endcaps
	edm::ESHandle<GBRForest> ecalRegEndcapH;
	gbrRcd.get(cfg_.ecalRegressionWeightLabels[1].c_str(),ecalRegEndcapH);
	ecalRegEndcap_= &(*ecalRegEndcapH);
	
	// ECAL barrel error
	edm::ESHandle<GBRForest> ecalRegErrorBarrelH;
	gbrRcd.get(cfg_.ecalRegressionWeightLabels[2].c_str(),ecalRegErrorBarrelH);
	ecalRegErrorBarrel_ = &(*ecalRegErrorBarrelH);
	
	// ECAL endcap error
	edm::ESHandle<GBRForest> ecalRegErrorEndcapH;
	gbrRcd.get(cfg_.ecalRegressionWeightLabels[3].c_str(),ecalRegErrorEndcapH);
	ecalRegErrorEndcap_ = &(*ecalRegErrorEndcapH);
	
	ecalRegressionInitialized_ = true;
      }
    }
  if( cfg_.combinationWeightsFromDB) 
    {

	// Combination
	if(cfg_.combinationWeightsFromDB) 
	  {
	    unsigned long long newRegressionCacheId
	      = es.get<GBRWrapperRcd>().cacheIdentifier() ;
	    if ( !regressionCacheId_ || (newRegressionCacheId != regressionCacheId_ ) ) {

	      const GBRWrapperRcd& gbrRcd = es.get<GBRWrapperRcd>();

	      edm::ESHandle<GBRForest> combinationRegH;
	      gbrRcd.get(cfg_.combinationRegressionWeightLabels[0].c_str(),combinationRegH);
	      combinationReg_ = &(*combinationRegH);
	      
	      combinationRegressionInitialized_ = true;
	    }
	  }
    }

  // read weights from file - for debugging. Even if it is one single files, 4 files should b set in the vector
  if(!cfg_.ecalWeightsFromDB && !ecalRegressionInitialized_ &&
     cfg_.ecalRegressionWeightFiles.size() ) {
    TFile file0 (edm::FileInPath(cfg_.ecalRegressionWeightFiles[0].c_str()).fullPath().c_str());
    ecalRegBarrel_ = (const GBRForest*)file0.Get(cfg_.ecalRegressionWeightLabels[0].c_str());
    file0.Close();
    TFile file1 (edm::FileInPath(cfg_.ecalRegressionWeightFiles[1].c_str()).fullPath().c_str());
    ecalRegEndcap_ = (const GBRForest*)file1.Get(cfg_.ecalRegressionWeightLabels[1].c_str());
    file1.Close();
    TFile file2 (edm::FileInPath(cfg_.ecalRegressionWeightFiles[2].c_str()).fullPath().c_str());
    ecalRegErrorBarrel_ = (const GBRForest*)file2.Get(cfg_.ecalRegressionWeightLabels[2].c_str());
    file2.Close();
    TFile file3 (edm::FileInPath(cfg_.ecalRegressionWeightFiles[3].c_str()).fullPath().c_str());
    ecalRegErrorEndcap_ = (const GBRForest*)file3.Get(cfg_.ecalRegressionWeightLabels[3].c_str());
    ecalRegressionInitialized_ = true;
    file3.Close();
  }
  
  if(!cfg_.combinationWeightsFromDB && !combinationRegressionInitialized_ && 
     cfg_.combinationRegressionWeightFiles.size() ) 
    {
      TFile file0 (edm::FileInPath(cfg_.combinationRegressionWeightFiles[0].c_str()).fullPath().c_str());
      combinationReg_ = (const GBRForest*)file0.Get(cfg_.combinationRegressionWeightLabels[0].c_str());
      combinationRegressionInitialized_ = true;
      file0.Close();
    }
  
}

void RegressionHelper::readEvent( const edm::Event &) {;}

void RegressionHelper::getEcalRegression(const reco::SuperCluster & sc,
					 const edm::Handle<reco::VertexCollection>& vertices,
					 const edm::Handle<EcalRecHitCollection>& rechitsEB,
					 const edm::Handle<EcalRecHitCollection>& rechitsEE,
					 double & energyFactor, double & errorFactor) const {
  
  energyFactor = -999.;
  errorFactor = -999.;
  std::vector<float> rInputs;
  rInputs.resize(33);
  
  const double rawEnergy = sc.rawEnergy();
  const double calibEnergy = sc.correctedEnergy();
  const edm::Ptr<reco::CaloCluster> seed(sc.seed());
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
  if ( seed->hitsAndFractions()[0].first.subdetId()==EcalBarrel ) {
    const float eMax = EcalClusterTools::eMax( *seed, &*rechitsEB );
    const float e2nd = EcalClusterTools::e2nd( *seed, &*rechitsEB );
    const float e3x3 = EcalClusterTools::e3x3( *seed,
					       &*rechitsEB,
					       &*caloTopology_ );
    const float eTop = EcalClusterTools::eTop( *seed,
					       &*rechitsEB,
					       &*caloTopology_ );
    const float eBottom = EcalClusterTools::eBottom( *seed,
						     &*rechitsEB,
						     &*caloTopology_ );
    const float eLeft = EcalClusterTools::eLeft( *seed,
                                                 &*rechitsEB,
                                                 &*caloTopology_ );
    const float eRight = EcalClusterTools::eRight( *seed,
						   &*rechitsEB,
						   &*caloTopology_ );
    const float eLpeR = eLeft + eRight;
    const float eTpeB = eTop + eBottom;
    const float eLmeR = eLeft - eRight;
    const float eTmeB = eTop - eBottom;
    std::vector<float> vCov =
      EcalClusterTools::localCovariances( *seed, &*rechitsEB, &*caloTopology_ );
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
    ecl_.localCoordsEB(*seed, *caloGeometry_, cryEta, cryPhi,
		       ieta, iphi, thetatilt, phitilt);
    rInputs[0] = nVtx; //nVtx
    rInputs[1] = sc.eta(); //scEta
    rInputs[2] = sc.phi(); //scPhi
    rInputs[3] = sc.etaWidth(); //scEtaWidth
    rInputs[4] = sc.phiWidth(); //scPhiWidth
    rInputs[5] = e3x3/rawEnergy; //scSeedR9
    rInputs[6] = sc.seed()->energy()/rawEnergy; //scSeedRawEnergy/scRawEnergy
    rInputs[7] = eMax/rawEnergy; //scSeedEmax/scRawEnergy
    rInputs[8] = e2nd/rawEnergy; //scSeedE2nd/scRawEnergy
    rInputs[9] = (eLpeR!=0. ? eLmeR/eLpeR : 0.);//scSeedLeftRightAsym
    rInputs[10] = (eTpeB!=0.? eTmeB/eTpeB : 0.);//scSeedTopBottomAsym
    rInputs[11] = see; //scSeedSigmaIetaIeta
    rInputs[12] = sep; //scSeedSigmaIetaIphi
    rInputs[13] = spp; //scSeedSigmaIphiIphi
    rInputs[14] = sc.clustersSize()-1; //N_ECALClusters
    rInputs[15] = maxDR; //clusterMaxDR
    rInputs[16] = maxDRDPhi; //clusterMaxDRDPhi
    rInputs[17] = maxDRDEta; //clusterMaxDRDEta
    rInputs[18] = maxDRRawEnergy/rawEnergy; //clusMaxDRRawEnergy/scRawEnergy
    rInputs[19] = subClusRawE[0]/rawEnergy; //clusterRawEnergy[0]/scRawEnergy
    rInputs[20] = subClusRawE[1]/rawEnergy; //clusterRawEnergy[1]/scRawEnergy
    rInputs[21] = subClusRawE[2]/rawEnergy; //clusterRawEnergy[2]/scRawEnergy
    rInputs[22] = subClusDPhi[0]; //clusterDPhiToSeed[0]
    rInputs[23] = subClusDPhi[1]; //clusterDPhiToSeed[1]
    rInputs[24] = subClusDPhi[2]; //clusterDPhiToSeed[2]
    rInputs[25] = subClusDEta[0]; //clusterDEtaToSeed[0]
    rInputs[26] = subClusDEta[1]; //clusterDEtaToSeed[1]
    rInputs[27] = subClusDEta[2]; //clusterDEtaToSeed[2]
    rInputs[28] = cryEta; //scSeedCryEta
    rInputs[29] = cryPhi; //scSeedCryPhi
    rInputs[30] = ieta; //scSeedCryIeta
    rInputs[31] = iphi; //scSeedCryIphi
    rInputs[32] = calibEnergy; //scCalibratedEnergy
    energyFactor = ecalRegBarrel_->GetResponse(&rInputs[0]);
    errorFactor = ecalRegErrorBarrel_->GetResponse(&rInputs[0]);
  }
  else if(seed->hitsAndFractions()[0].first.subdetId()==EcalEndcap)
    {
      const float eMax = EcalClusterTools::eMax( *seed, &*rechitsEE );
      const float e2nd = EcalClusterTools::e2nd( *seed, &*rechitsEE );
      const float e3x3 = EcalClusterTools::e3x3( *seed,
                                                 &*rechitsEE,
                                                 &*caloTopology_ );
      const float eTop = EcalClusterTools::eTop( *seed,
                                                 &*rechitsEE,
                                                 &*caloTopology_ );
      const float eBottom = EcalClusterTools::eBottom( *seed,
						       &*rechitsEE,
						       &*caloTopology_ );
      const float eLeft = EcalClusterTools::eLeft( *seed,
						   &*rechitsEE,
						   &*caloTopology_ );
      const float eRight = EcalClusterTools::eRight( *seed,
						     &*rechitsEE,
						     &*caloTopology_ );
      const float eLpeR = eLeft + eRight;
      const float eTpeB = eTop + eBottom;
      const float eLmeR = eLeft - eRight;
      const float eTmeB = eTop - eBottom;
      std::vector<float> vCov =
        EcalClusterTools::localCovariances( *seed, &*rechitsEE, &*caloTopology_ );
      const float see = (isnan(vCov[0]) ? 0. : sqrt(vCov[0]));
      const float spp = (isnan(vCov[2]) ? 0. : sqrt(vCov[2]));
      float sep = 0.;
      if (see*spp > 0)
        sep = vCov[1] / (see * spp);
      else if (vCov[1] > 0)
        sep = 1.0;
      else
        sep = -1.0;
      rInputs[0] = nVtx; //nVtx
      rInputs[1] = sc.eta(); //scEta
      rInputs[2] = sc.phi(); //scPhi
      rInputs[3] = sc.etaWidth(); //scEtaWidth
      rInputs[4] = sc.phiWidth(); //scPhiWidth
      rInputs[5] = e3x3/rawEnergy; //scSeedR9
      rInputs[6] = sc.seed()->energy()/rawEnergy; //scSeedRawEnergy/scRawEnergy
      rInputs[7] = eMax/rawEnergy; //scSeedEmax/scRawEnergy
      rInputs[8] = e2nd/rawEnergy; //scSeedE2nd/scRawEnergy
      rInputs[9] = (eLpeR!=0. ? eLmeR/eLpeR : 0.);//scSeedLeftRightAsym
      rInputs[10] = (eTpeB!=0.? eTmeB/eTpeB : 0.);//scSeedTopBottomAsym
      rInputs[11] = see; //scSeedSigmaIetaIeta
      rInputs[12] = sep; //scSeedSigmaIetaIphi
      rInputs[13] = spp; //scSeedSigmaIphiIphi
      rInputs[14] = sc.clustersSize()-1; //N_ECALClusters
      rInputs[15] = maxDR; //clusterMaxDR
      rInputs[16] = maxDRDPhi; //clusterMaxDRDPhi
      rInputs[17] = maxDRDEta; //clusterMaxDRDEta
      rInputs[18] = maxDRRawEnergy/rawEnergy; //clusMaxDRRawEnergy/scRawEnergy
      rInputs[19] = subClusRawE[0]/rawEnergy; //clusterRawEnergy[0]/scRawEnergy
      rInputs[20] = subClusRawE[1]/rawEnergy; //clusterRawEnergy[1]/scRawEnergy
      rInputs[21] = subClusRawE[2]/rawEnergy; //clusterRawEnergy[2]/scRawEnergy
      rInputs[22] = subClusDPhi[0]; //clusterDPhiToSeed[0]
      rInputs[23] = subClusDPhi[1]; //clusterDPhiToSeed[1]
      rInputs[24] = subClusDPhi[2]; //clusterDPhiToSeed[2]
      rInputs[25] = subClusDEta[0]; //clusterDEtaToSeed[0]
      rInputs[26] = subClusDEta[1]; //clusterDEtaToSeed[1]
      rInputs[27] = subClusDEta[2]; //clusterDEtaToSeed[2]
      rInputs[28] = scPreshowerSum/rawEnergy; //scPreshowerEnergy/scRawEnergy
      rInputs[29] = calibEnergy; //scCalibratedEnergy
      energyFactor = ecalRegEndcap_->GetResponse(&rInputs[0]);
      errorFactor = ecalRegErrorEndcap_->GetResponse(&rInputs[0]);
    }
  else
    {
      //      std::cout << " SubdetID " << " " << seed->seed().rawId() << " " << seed->seed().subdetId() << std::endl;
      throw cms::Exception("RegressionHelper::calculateRegressedEnergy")
	<< "Supercluster seed is either EB nor EE!" << std::endl;
    }
}

void RegressionHelper::applyCombinationRegression(reco::GsfElectron & ele) const
{
  float energy = ele.correctedEcalEnergy();
  float energyError = ele.correctedEcalEnergyError();
  float momentum = ele.trackMomentumAtVtx().R();
  float momentumError = ele.trackMomentumError();
  int elClass = -.1;
  
  switch (ele.classification()) {
  case reco::GsfElectron::GOLDEN:
    elClass = 0;
    break;
  case reco::GsfElectron::BIGBREM:
    elClass = 1;
    break;
  case reco::GsfElectron::BADTRACK:
    elClass = 2;
    break;
  case  reco::GsfElectron::SHOWERING:
    elClass = 3;
    break;
  case reco::GsfElectron::GAP:
    elClass = 4;
    break;
  default:
    elClass = -1;


  bool isEcalDriven = ele.ecalDriven();
  bool isTrackerDriven =  ele.trackerDrivenSeed();
  bool isEB = ele.isEB();

  // compute relative errors and ratio of errors
  float energyRelError = energyError / energy;
  float momentumRelError = momentumError / momentum;
  float errorRatio = energyRelError / momentumRelError;

  // calculate E/p and corresponding error
  float eOverP = energy / momentum;  
  float eOverPerror = eOverP*std::hypot(energyRelError,momentumRelError);
  
  // fill input variables
  std::vector<float> regressionInputs ;
  regressionInputs.resize(11,0.);

  regressionInputs[0]  = energy;
  regressionInputs[1]  = energyRelError;
  regressionInputs[2]  = momentum;
  regressionInputs[3]  = momentumRelError;
  regressionInputs[4]  = errorRatio;
  regressionInputs[5]  = eOverP;
  regressionInputs[6]  = eOverPerror;
  regressionInputs[7]  = static_cast<float>(isEcalDriven);
  regressionInputs[8]  = static_cast<float>(isTrackerDriven);
  regressionInputs[9]  = static_cast<float>(elClass);
  regressionInputs[10] = static_cast<float>(isEB);
    
  // retrieve combination weight
  float weight = 0.;
  if(eOverP>0.025 
     &&fabs(momentum-energy)<15.*sqrt(momentumError*momentumError + energyError*energyError)
     ) // protection against crazy track measurement
    {
      weight = combinationReg_->GetResponse(regressionInputs.data());
      if(weight>1.) weight = 1.;
      else if(weight<0.) weight = 0.;
    }

  float combinedMomentum = weight*momentum + (1.-weight)*energy;
  float combinedMomentumError = sqrt(weight*weight*momentumError*momentumError + (1.-weight)*(1.-weight)*energyError*energyError);

  // FIXME : pure tracker electrons have track momentum error of 999.
  // If the combination try to combine such electrons then the original combined momentum is kept
  if(momentumError!=999. || weight==0.)
    {
      math::XYZTLorentzVector oldMomentum = ele.p4() ;
      math::XYZTLorentzVector newMomentum(oldMomentum.x()*combinedMomentum/oldMomentum.t(),
					  oldMomentum.y()*combinedMomentum/oldMomentum.t(),
					  oldMomentum.z()*combinedMomentum/oldMomentum.t(),
					  combinedMomentum);
					  
      ele.setP4(reco::GsfElectron::P4_COMBINATION,newMomentum,combinedMomentumError,true);
    }
  }
}
