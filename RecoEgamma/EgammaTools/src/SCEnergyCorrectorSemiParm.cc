//#include <TFile.h>
#include "../interface/SCEnergyCorrectorSemiParm.h"
#include "CondFormats/EgammaObjects/interface/GBRForestD.h"
#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"
#include "FWCore/Framework/interface/ESHandle.h" 
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RooWorkspace.h"
#include "RooArgList.h"
#include "RooRealVar.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooConstVar.h"
#include "TStreamerInfo.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include <vdt/vdtMath.h>

using namespace reco;

//--------------------------------------------------------------------------------------------------
SCEnergyCorrectorSemiParm::SCEnergyCorrectorSemiParm() :
foresteb_(0),
forestee_(0),
forestsigmaeb_(0),
forestsigmaee_(0),
calotopo_(0),
calogeom_(0),
topo_record_(0),
geom_record_(0)
{}

//--------------------------------------------------------------------------------------------------
SCEnergyCorrectorSemiParm::~SCEnergyCorrectorSemiParm()
{}

//--------------------------------------------------------------------------------------------------
void SCEnergyCorrectorSemiParm::setTokens(const edm::ParameterSet &iConfig, edm::ConsumesCollector &cc) {
  
  isHLT_ = iConfig.getParameter<bool>("isHLT");

  tokenEBRecHits_   = cc.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ecalRecHitsEB"));
  tokenEERecHits_   = cc.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ecalRecHitsEE"));

  regressionKey_  = iConfig.getParameter<std::string>("regressionKey");
  uncertaintyKey_ = iConfig.getParameter<std::string>("uncertaintyKey");
 
  if (not isHLT_)
    tokenVertices_     = cc.consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexCollection"));
  else
    tokenRho_          = cc.consumes<double>(iConfig.getParameter<edm::InputTag>("rhoCollection"));
}

//--------------------------------------------------------------------------------------------------
void SCEnergyCorrectorSemiParm::setEventSetup(const edm::EventSetup &es) {
  
  const CaloTopologyRecord& topofrom_es = es.get<CaloTopologyRecord>();
  if( !topo_record_ ||
      topofrom_es.cacheIdentifier() != topo_record_->cacheIdentifier() ) {
    topo_record_ = &topofrom_es;
    topo_record_->get(calotopo_);
  }
  const CaloGeometryRecord& geomfrom_es = es.get<CaloGeometryRecord>();
  if( !geom_record_ ||
      geomfrom_es.cacheIdentifier() != geom_record_->cacheIdentifier() ) {
    geom_record_ = &geomfrom_es;
    geom_record_->get(calogeom_);
  }  

  edm::ESHandle<GBRForestD> readereb;
  edm::ESHandle<GBRForestD> readerebvar;
  edm::ESHandle<GBRForestD> readeree;
  edm::ESHandle<GBRForestD> readereevar;
  
  es.get<GBRDWrapperRcd>().get(std::string(TString::Format("%s_EBCorrection",  regressionKey_.c_str())), readereb);
  es.get<GBRDWrapperRcd>().get(std::string(TString::Format("%s_EBUncertainty", uncertaintyKey_.c_str())), readerebvar);
  es.get<GBRDWrapperRcd>().get(std::string(TString::Format("%s_EECorrection",  regressionKey_.c_str())), readeree);
  es.get<GBRDWrapperRcd>().get(std::string(TString::Format("%s_EEUncertainty", uncertaintyKey_.c_str())), readereevar);
  
  foresteb_      = readereb.product();
  forestsigmaeb_ = readerebvar.product();
  forestee_      = readeree.product();
  forestsigmaee_ = readereevar.product();
}

//--------------------------------------------------------------------------------------------------
void SCEnergyCorrectorSemiParm::setEvent(const edm::Event &e) {
  
  e.getByToken(tokenEBRecHits_,rechitsEB_);
  e.getByToken(tokenEERecHits_,rechitsEE_);

  if (not isHLT_)
    e.getByToken(tokenVertices_,vertices_);
  else
    e.getByToken(tokenRho_, rho_);
}

//--------------------------------------------------------------------------------------------------
void SCEnergyCorrectorSemiParm::modifyObject(reco::SuperCluster &sc) {
  
  const reco::CaloCluster &seedCluster = *(sc.seed());
  const bool iseb = seedCluster.hitsAndFractions()[0].first.subdetId() == EcalBarrel;
  const EcalRecHitCollection *recHits = iseb ? rechitsEB_.product() : rechitsEE_.product();

  const CaloTopology *topo = calotopo_.product();
  
  const double raw_energy = sc.rawEnergy();   
  const int numberOfClusters =  sc.clusters().size();

  std::vector<float> localCovariances = EcalClusterTools::localCovariances(seedCluster,recHits,topo) ;
  
  if (not isHLT_) {
  
    std::array<float, 29> eval;  
    
    const float eLeft = EcalClusterTools::eLeft(seedCluster,recHits,topo);
    const float eRight = EcalClusterTools::eRight(seedCluster,recHits,topo);
    const float eTop = EcalClusterTools::eTop(seedCluster,recHits,topo);
    const float eBottom = EcalClusterTools::eBottom(seedCluster,recHits,topo);
    
    float sigmaIetaIeta = sqrt(localCovariances[0]);
    float sigmaIetaIphi = std::numeric_limits<float>::max();
    float sigmaIphiIphi = std::numeric_limits<float>::max();
    
    // extra shower shapes
    const float see_by_spp = sigmaIetaIeta*sigmaIphiIphi;
    if(  see_by_spp > 0 ) {
      sigmaIetaIphi = localCovariances[1] / see_by_spp;
    } else if ( localCovariances[1] > 0 ) {
      sigmaIetaIphi = 1.f;
    } else {
      sigmaIetaIphi = -1.f;
    }
    
    if (!edm::isNotFinite(localCovariances[2])) sigmaIphiIphi = sqrt(localCovariances[2]) ;
    
    // calculate sub-cluster variables
    std::vector<float> clusterRawEnergy;
    clusterRawEnergy.resize(std::max(3, numberOfClusters), 0);
    std::vector<float> clusterDEtaToSeed;
    clusterDEtaToSeed.resize(std::max(3, numberOfClusters), 0);
    std::vector<float> clusterDPhiToSeed;
    clusterDPhiToSeed.resize(std::max(3, numberOfClusters), 0);
    float clusterMaxDR     = 999.;
    float clusterMaxDRDPhi = 999.;
    float clusterMaxDRDEta = 999.;
    float clusterMaxDRRawEnergy = 0.;
    
    size_t iclus = 0;
    float maxDR = 0;
    edm::Ptr<reco::CaloCluster> pclus;
    const edm::Ptr<reco::CaloCluster>& theseed = sc.seed();
    // loop over all clusters that aren't the seed  
    auto clusend = sc.clustersEnd();
    for( auto clus = sc.clustersBegin(); clus != clusend; ++clus ) {
      pclus = *clus;
      
      if(theseed == pclus ) 
	continue;
      clusterRawEnergy[iclus]  = pclus->energy();
      clusterDPhiToSeed[iclus] = reco::deltaPhi(pclus->phi(),theseed->phi());
      clusterDEtaToSeed[iclus] = pclus->eta() - theseed->eta();
      
      // find cluster with max dR
      const auto the_dr = reco::deltaR(*pclus, *theseed);
      if(the_dr > maxDR) {
	maxDR = the_dr;
	clusterMaxDR = maxDR;
	clusterMaxDRDPhi = clusterDPhiToSeed[iclus];
	clusterMaxDRDEta = clusterDEtaToSeed[iclus];
	clusterMaxDRRawEnergy = clusterRawEnergy[iclus];
      }      
      ++iclus;
    }  
    
    // SET INPUTS
    eval[0]  = vertices_->size(); //nVtx
    eval[1]  = raw_energy; //scRawEnergy
    eval[2]  = sc.etaWidth(); //scEtaWidth
    eval[3]  = sc.phiWidth(); //scPhiWidth
    eval[4]  = EcalClusterTools::e3x3(seedCluster,recHits,topo)/raw_energy; //scSeedR9
    eval[5]  = seedCluster.energy()/raw_energy; //scSeedRawEnergy/scRawEnergy
    eval[6]  = EcalClusterTools::eMax(seedCluster,recHits)/raw_energy;  //scSeedEmax/scRawEnergy
    eval[7]  = EcalClusterTools::e2nd(seedCluster,recHits)/raw_energy; // scSeedE2nd/scRawEnergy
    eval[8] = (eLeft + eRight != 0.f  ? (eLeft-eRight)/(eLeft+eRight) : 0.f); //scSeedLeftRightAsym
    eval[9] = (eTop  + eBottom != 0.f ? (eTop-eBottom)/(eTop+eBottom) : 0.f); //scSeedTopBottomAsym
    eval[10] = sigmaIetaIeta; //scSeedSigmaIetaIeta
    eval[11] = sigmaIetaIphi; //scSeedSigmaIetaIphi
    eval[12] = sigmaIphiIphi; //scSeedSigmaIphiIphi
    eval[13] = std::max(0,numberOfClusters-1); //N_ECALClusters
    eval[14] = clusterMaxDR; //clusterMaxDR
    eval[15] = clusterMaxDRDPhi; //clusterMaxDRDPhi
    eval[16] = clusterMaxDRDEta; //clusterMaxDRDEta
    eval[17] = clusterMaxDRRawEnergy/raw_energy; //clusterMaxDRRawEnergy/scRawEnergy
    eval[18] = clusterRawEnergy[0]/raw_energy; //clusterRawEnergy[0]/scRawEnergy
    eval[19] = clusterRawEnergy[1]/raw_energy; //clusterRawEnergy[1]/scRawEnergy
    eval[20] = clusterRawEnergy[2]/raw_energy; //clusterRawEnergy[2]/scRawEnergy
    eval[21] = clusterDPhiToSeed[0]; //clusterDPhiToSeed[0]
    eval[22] = clusterDPhiToSeed[1]; //clusterDPhiToSeed[1]
    eval[23] = clusterDPhiToSeed[2]; //clusterDPhiToSeed[2]
    eval[24] = clusterDEtaToSeed[0]; //clusterDEtaToSeed[0]
    eval[25] = clusterDEtaToSeed[1]; //clusterDEtaToSeed[1]
    eval[26] = clusterDEtaToSeed[2]; //clusterDEtaToSeed[2]
    if (iseb) {
      EBDetId ebseedid(seedCluster.seed());
      eval[27] = ebseedid.ieta(); //scSeedCryIetaV2
      eval[28] = ebseedid.iphi(); //scSeedCryIphiV2
    } else {
      EEDetId eeseedid(seedCluster.seed());
      eval[27] = eeseedid.ix(); //scSeedCryIxV2
      eval[28] = eeseedid.iy(); //scSeedCryIyV2
    }  
    
    //magic numbers for MINUIT-like transformation of BDT output onto limited range
    //(These should be stored inside the conditions object in the future as well)
    constexpr double meanlimlow  = 0.2;
    constexpr double meanlimhigh = 2.0;
    constexpr double meanoffset  = meanlimlow + 0.5*(meanlimhigh-meanlimlow);
    constexpr double meanscale   = 0.5*(meanlimhigh-meanlimlow);
    
    constexpr double sigmalimlow  = 0.0002;
    constexpr double sigmalimhigh = 0.5;
    constexpr double sigmaoffset  = sigmalimlow + 0.5*(sigmalimhigh-sigmalimlow);
    constexpr double sigmascale   = 0.5*(sigmalimhigh-sigmalimlow);  
    
    const GBRForestD *forestmean = iseb ? foresteb_ : forestee_;
    const GBRForestD *forestsigma = iseb ? forestsigmaeb_ : forestsigmaee_;
    
    //these are the actual BDT responses
    double rawmean = forestmean->GetResponse(eval.data());
    double rawsigma = forestsigma->GetResponse(eval.data());
    
    //apply transformation to limited output range (matching the training)
    double mean = meanoffset + meanscale*vdt::fast_sin(rawmean);
    double sigma = sigmaoffset + sigmascale*vdt::fast_sin(rawsigma);
    
    double ecor = mean*(eval[1]);
    const double sigmacor = sigma*ecor;
    
    sc.setEnergy(ecor);
    sc.setCorrectedEnergy(ecor);
    sc.setCorrectedEnergyUncertainty(sigmacor);
  } else {

    std::array<float, 7> eval;  
    
    float clusterMaxDR     = 999.;
    size_t iclus = 0;
    float maxDR = 0;
    edm::Ptr<reco::CaloCluster> pclus;
    const edm::Ptr<reco::CaloCluster>& theseed = sc.seed();

    // loop over all clusters that aren't the seed  
    auto clusend = sc.clustersEnd();
    for( auto clus = sc.clustersBegin(); clus != clusend; ++clus ) {
      pclus = *clus;
      
      if(theseed == pclus ) 
	continue;
      
      // find cluster with max dR
      const auto the_dr = reco::deltaR(*pclus, *theseed);
      if(the_dr > maxDR) {
	maxDR = the_dr;
	clusterMaxDR = maxDR;
      }      
      ++iclus;
    }  
    
    // SET INPUTS
    eval[0] = *(rho_.product());
    eval[1] = sc.eta(); 
    eval[2] = sc.phiWidth(); 
    eval[3] = EcalClusterTools::e3x3(seedCluster,recHits,topo)/raw_energy;
    eval[4] = std::max(0,numberOfClusters-1);
    eval[5] = clusterMaxDR;
    eval[6] = raw_energy;
    
    //magic numbers for MINUIT-like transformation of BDT output onto limited range
    //(These should be stored inside the conditions object in the future as well)
    constexpr double meanlimlow  = 0.2;
    constexpr double meanlimhigh = 2.0;
    constexpr double meanoffset  = meanlimlow + 0.5*(meanlimhigh-meanlimlow);
    constexpr double meanscale   = 0.5*(meanlimhigh-meanlimlow);
    
    constexpr double sigmalimlow  = 0.0002;
    constexpr double sigmalimhigh = 0.5;
    constexpr double sigmaoffset  = sigmalimlow + 0.5*(sigmalimhigh-sigmalimlow);
    constexpr double sigmascale   = 0.5*(sigmalimhigh-sigmalimlow);  
    
    const GBRForestD *forestmean = iseb ? foresteb_ : forestee_;
    const GBRForestD *forestsigma = iseb ? forestsigmaeb_ : forestsigmaee_;
    
    //these are the actual BDT responses
    double rawmean = forestmean->GetResponse(eval.data());
    double rawsigma = forestsigma->GetResponse(eval.data());
    
    //apply transformation to limited output range (matching the training)
    double mean = meanoffset + meanscale*vdt::fast_sin(rawmean);
    double sigma = sigmaoffset + sigmascale*vdt::fast_sin(rawsigma);

    double ecor = mean*eval[6];
    if (!iseb)  
      ecor = mean*(eval[6]+sc.preshowerEnergy());
    const double sigmacor = sigma*ecor;
   
    sc.setEnergy(ecor);
    sc.setCorrectedEnergy(ecor);
    sc.setCorrectedEnergyUncertainty(sigmacor);
  }  
}

