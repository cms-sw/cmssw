#include "RecoEcal/EgammaCoreTools/interface/EcalClustersGraph.h"
#include <algorithm>
#include <cmath>
#include "TVector2.h"
#include "TMath.h"
#include <iostream>

using namespace std;
using namespace reco;

typedef std::shared_ptr<CalibratedPFCluster> CalibratedClusterPtr;
typedef std::vector<CalibratedClusterPtr> CalibratedClusterPtrVector;

EcalClustersGraph::EcalClustersGraph(CalibratedClusterPtrVector clusters,
                                     int nSeeds,
                                     const CaloTopology* topology,
                                     const CaloSubdetectorGeometry* ebGeom,
                                     const CaloSubdetectorGeometry* eeGeom,
                                     const EcalRecHitCollection* recHitsEB,
                                     const EcalRecHitCollection* recHitsEE,
                                     const SCProducerCache* cache)
    : clusters_(clusters),
      nSeeds_(nSeeds),
      topology_(topology),
      ebGeom_(ebGeom),
      eeGeom_(eeGeom),
      recHitsEB_(recHitsEB),
      recHitsEE_(recHitsEE),
      SCProducerCache_(cache) {
  nCls_ = clusters_.size();
  inWindows_ = GraphMatrix<int>(nSeeds_, nCls_);
  scoreMatrix_ = GraphMatrix<float>(nSeeds_, nCls_);
  clusterMatrix_ = GraphMatrix<float>(nSeeds_, nCls_);
  Rnd = new TRandom();

  // Prepare the batch size of the tensor inputs == number of windows
  inputs_.clustersX.resize(nSeeds_);
  inputs_.windowX.resize(nSeeds_);
  inputs_.hitsX.resize(nSeeds_);
  inputs_.isSeed.resize(nSeeds_);

  LogDebug("EcalClustersGraph") << "EcalClustersGraph created. nSeeds " << nSeeds_ << ", nClusters " << nCls_ << endl;
}

std::vector<int> EcalClustersGraph::clusterPosition(const CaloCluster* cluster) {
  std::vector<int> coordinates;
  coordinates.resize(3);
  int ieta = -999;
  int iphi = -999;
  int iz = -99;

  math::XYZPoint caloPos = cluster->position();
  if (PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_BARREL) {
    EBDetId eb_id(ebGeom_->getClosestCell(GlobalPoint(caloPos.x(), caloPos.y(), caloPos.z())));
    ieta = eb_id.ieta();
    iphi = eb_id.iphi();
    iz = 0;
  } else if (PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_ENDCAP) {
    EEDetId ee_id(eeGeom_->getClosestCell(GlobalPoint(caloPos.x(), caloPos.y(), caloPos.z())));
    ieta = ee_id.ix();
    iphi = ee_id.iy();
    if (ee_id.zside() < 0)
      iz = -1;
    if (ee_id.zside() > 0)
      iz = 1;
  }

  coordinates[0] = ieta;
  coordinates[1] = iphi;
  coordinates[2] = iz;
  return coordinates;
}

std::vector<double> EcalClustersGraph::dynamicWindow(double seedEta) {
  std::vector<double> window;
  window.resize(3);

  double eta = fabs(seedEta);
  double deta_down = 0.;
  double deta_up = 0.;
  double dphi = 0.;

  //deta_down
  if (eta < 2.1)
    deta_down = -0.075;
  else if (eta >= 2.1 && eta < 2.5)
    deta_down = -0.1875 * eta + 0.31875;
  else if (eta >= 2.5)
    deta_down = -0.15;

  //deta_up
  if (eta >= 0 && eta < 0.1)
    deta_up = 0.075;
  else if (eta >= 0.1 && eta < 1.3)
    deta_up = 0.0758929 - 0.0178571 * eta + 0.0892857 * (eta * eta);
  else if (eta >= 1.3 && eta < 1.7)
    deta_up = 0.2;
  else if (eta >= 1.7 && eta < 1.9)
    deta_up = 0.625 - 0.25 * eta;
  else if (eta >= 1.9)
    deta_up = 0.15;

  //dphi
  if (eta < 1.9)
    dphi = 0.6;
  else if (eta >= 1.9 && eta < 2.7)
    dphi = 1.075 - 0.25 * eta;
  else if (eta >= 2.7)
    dphi = 0.4;

  window[0] = deta_down;
  window[1] = deta_up;
  window[2] = dphi;

  return window;
}

void EcalClustersGraph::initWindows() {
  for (uint is = 0; is < nSeeds_; is++) {
    std::vector<int> seedLocal = clusterPosition((*clusters_.at(is)).the_ptr().get());
    double seed_eta = clusters_.at(is)->eta();
    double seed_phi = clusters_.at(is)->phi();
    inWindows_.Set(is, is, 1);
    std::vector<double> width = dynamicWindow(seed_eta);

    for (uint icl = is + 1; icl < nCls_; icl++) {
      std::vector<int> clusterLocal = clusterPosition((*clusters_.at(icl)).the_ptr().get());
      double cl_eta = clusters_.at(icl)->eta();
      double cl_phi = clusters_.at(icl)->phi();
      double dphi = deltaPhi(seed_phi, cl_phi);
      double deta = deltaEta(seed_eta, cl_eta);

      int isIn = 0;
      if (seedLocal[2] == clusterLocal[2] && deta >= width[0] && deta <= width[1] && fabs(dphi) <= width[2])
        isIn = 1;

      inWindows_.Set(is, icl, isIn);
      //Save also symmetric part of the adj matrix
      if (icl < nSeeds_)
        inWindows_.Set(icl, is, isIn);
    }
  }
}

void EcalClustersGraph::clearWindows() {
  inWindows_.Clear();
  scoreMatrix_.Clear();
  clusterMatrix_.Clear();
}

std::pair<double,double> EcalClustersGraph::computeCovariances(const CaloCluster* cluster) 
{

     double etaWidth = 0.;
     double phiWidth = 0.;
     double numeratorEtaWidth = 0;
     double numeratorPhiWidth = 0;

     double clEnergy = cluster->energy();
     double denominator = clEnergy;

     double clEta = cluster->position().eta();
     double clPhi = cluster->position().phi();

     std::shared_ptr<const CaloCellGeometry> this_cell;
     EcalRecHitCollection::const_iterator rHit;

     const std::vector<std::pair<DetId, float> >& detId = cluster->hitsAndFractions();
     // Loop over recHits associated with the given SuperCluster
     for (std::vector<std::pair<DetId, float> >::const_iterator hit = detId.begin(); hit != detId.end(); ++hit) {     
       if(PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_BARREL){
          rHit = recHitsEB_->find((*hit).first);
          //FIXME: THIS IS JUST A WORKAROUND A FIX SHOULD BE APPLIED
          if (rHit == recHitsEB_->end()) {
              continue;
          }  
       }else if(PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_ENDCAP){
          rHit = recHitsEE_->find((*hit).first);
          //FIXME: THIS IS JUST A WORKAROUND A FIX SHOULD BE APPLIED
          if (rHit == recHitsEE_->end()) {
              continue;
          }  
       }  

       if(PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_BARREL){
          this_cell = ebGeom_->getGeometry(rHit->id());
       }else if(PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_ENDCAP){
          this_cell = eeGeom_->getGeometry(rHit->id());
       } 
       if (this_cell == nullptr) {
         //edm::LogInfo("SuperClusterShapeAlgo") << "pointer to the cell in Calculate_Covariances is NULL!";
         continue;
       }

       GlobalPoint position = this_cell->getPosition();
       //take into account energy fractions
       double energyHit = rHit->energy() * hit->second; 

       //form differences
       double dPhi = position.phi() - clPhi;
       if (dPhi > +Geom::pi()) {
         dPhi = Geom::twoPi() - dPhi;
       }
       if (dPhi < -Geom::pi()) {
         dPhi = Geom::twoPi() + dPhi;
       }

       double dEta = position.eta() - clEta;

       if (energyHit > 0) {
         numeratorEtaWidth += energyHit * dEta * dEta;
         numeratorPhiWidth += energyHit * dPhi * dPhi;
       }

       etaWidth = sqrt(numeratorEtaWidth / denominator);
       phiWidth = sqrt(numeratorPhiWidth / denominator);
     }

     return std::make_pair(etaWidth,phiWidth);
} 

std::vector<double> EcalClustersGraph::computeShowerShapes(const CaloCluster* cluster, bool full5x5=false)
{
     std::vector<double> showerVars_;
     showerVars_.resize(8);
     float e1=1.; 
     float e4=0.;
 
     if(full5x5){ 
        if(PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_BARREL){ 
           locCov_ = noZS::EcalClusterTools::localCovariances(*cluster, recHitsEB_, topology_);
           widths_ = computeCovariances(cluster);  
           e1 = noZS::EcalClusterTools::eMax(*cluster, recHitsEB_);
           e4 = noZS::EcalClusterTools::eTop(*cluster, recHitsEB_, topology_) +
                noZS::EcalClusterTools::eRight(*cluster, recHitsEB_, topology_) +
                noZS::EcalClusterTools::eBottom(*cluster, recHitsEB_, topology_) +
                noZS::EcalClusterTools::eLeft(*cluster, recHitsEB_, topology_); 
           showerVars_[0] = noZS::EcalClusterTools::e3x3(*cluster, recHitsEB_, topology_)/cluster->energy(); //r9
           showerVars_[1] = sqrt(locCov_[0]); //sigmaietaieta
           showerVars_[2] = locCov_[1]; //sigmaietaiphi
           showerVars_[3] = (!edm::isFinite(locCov_[2])) ? 0. : sqrt(locCov_[2]); //sigmaiphiiphi     
           showerVars_[4] = (e1!=0.) ? 1.-e4/e1 : -999.; //swiss_cross      
           showerVars_[5] = cluster->hitsAndFractions().size(); //nXtals 
           showerVars_[6] = widths_.first; //etaWidth 
           showerVars_[7] = widths_.second; //phiWidth   
        }else if(PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_ENDCAP){
           locCov_ = noZS::EcalClusterTools::localCovariances(*cluster, recHitsEE_, topology_);
           widths_ = computeCovariances(cluster);  
           e1 = noZS::EcalClusterTools::eMax(*cluster, recHitsEE_);
           e4 = noZS::EcalClusterTools::eTop(*cluster, recHitsEE_, topology_) +
                noZS::EcalClusterTools::eRight(*cluster, recHitsEE_, topology_) +
                noZS::EcalClusterTools::eBottom(*cluster, recHitsEE_, topology_) +
                noZS::EcalClusterTools::eLeft(*cluster, recHitsEE_, topology_); 
           showerVars_[0] = noZS::EcalClusterTools::e3x3(*cluster, recHitsEE_, topology_)/cluster->energy(); //r9
           showerVars_[1] = sqrt(locCov_[0]); //sigmaietaieta
           showerVars_[2] = locCov_[1]; //sigmaietaiphi
           showerVars_[3] = (!edm::isFinite(locCov_[2])) ? 0. : sqrt(locCov_[2]); //sigmaiphiiphi    
           showerVars_[4] = (e1!=0.) ? 1.-e4/e1 : -999.; //swiss_cross      
           showerVars_[5] = cluster->hitsAndFractions().size(); //nXtals 
           showerVars_[6] = widths_.first; //etaWidth 
           showerVars_[7] = widths_.second; //phiWidth  
        }
     }else{
        if(PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_BARREL){ 
           locCov_ = EcalClusterTools::localCovariances(*cluster, recHitsEB_, topology_);
           widths_ = computeCovariances(cluster);  
           e1 = EcalClusterTools::eMax(*cluster, recHitsEB_);
           e4 = EcalClusterTools::eTop(*cluster, recHitsEB_, topology_) +
                EcalClusterTools::eRight(*cluster, recHitsEB_, topology_) +
                EcalClusterTools::eBottom(*cluster, recHitsEB_, topology_) +
                EcalClusterTools::eLeft(*cluster, recHitsEB_, topology_); 
           showerVars_[0] = EcalClusterTools::e3x3(*cluster, recHitsEB_, topology_)/cluster->energy(); //r9
           showerVars_[1] = sqrt(locCov_[0]); //sigmaietaieta
           showerVars_[2] = locCov_[1]; //sigmaietaiphi
           showerVars_[3] = (!edm::isFinite(locCov_[2])) ? 0. : sqrt(locCov_[2]); //sigmaiphiiphi     
           showerVars_[4] = (e1!=0.) ? 1.-e4/e1 : -999.; //swiss_cross      
           showerVars_[5] = cluster->hitsAndFractions().size(); //nXtals 
           showerVars_[6] = widths_.first; //etaWidth 
           showerVars_[7] = widths_.second; //phiWidth   
        }else if(PFLayer::fromCaloID(cluster->caloID()) == PFLayer::ECAL_ENDCAP){
           locCov_ = EcalClusterTools::localCovariances(*cluster, recHitsEE_, topology_);
           widths_ = computeCovariances(cluster);  
           e1 = EcalClusterTools::eMax(*cluster, recHitsEE_);
           e4 = EcalClusterTools::eTop(*cluster, recHitsEE_, topology_) +
                EcalClusterTools::eRight(*cluster, recHitsEE_, topology_) +
                EcalClusterTools::eBottom(*cluster, recHitsEE_, topology_) +
                EcalClusterTools::eLeft(*cluster, recHitsEE_, topology_); 
           showerVars_[0] = EcalClusterTools::e3x3(*cluster, recHitsEE_, topology_)/cluster->energy(); //r9
           showerVars_[1] = sqrt(locCov_[0]); //sigmaietaieta
           showerVars_[2] = locCov_[1]; //sigmaietaiphi
           showerVars_[3] = (!edm::isFinite(locCov_[2])) ? 0. : sqrt(locCov_[2]); //sigmaiphiiphi    
           showerVars_[4] = (e1!=0.) ? 1.-e4/e1 : -999.; //swiss_cross      
           showerVars_[5] = cluster->hitsAndFractions().size(); //nXtals 
           showerVars_[6] = widths_.first; //etaWidth 
           showerVars_[7] = widths_.second; //phiWidth  
        }
     }

     return showerVars_; 
}

std::vector<std::vector<double>> EcalClustersGraph::fillHits(const CaloCluster* cluster) {
  const std::vector<std::pair<DetId, float>>& hitsAndFractions = cluster->hitsAndFractions();
  std::vector<std::vector<double>> out (hitsAndFractions.size());
  if (hitsAndFractions.size()==0){
    edm::LogError("EcalClustersGraph") << "No hits in cluster!!";
  }
  for (unsigned int i = 0; i < hitsAndFractions.size(); i++) {
    std::vector<double> rechit (DeepSCConfiguration::nRechitsFeatures);
    if (hitsAndFractions[i].first.subdetId() == EcalBarrel) {
      double energy = (*recHitsEB_->find(hitsAndFractions[i].first)).energy();
      EBDetId eb_id(hitsAndFractions[i].first);
      rechit[0] = eb_id.ieta();  //ieta
      rechit[1] = eb_id.iphi();  //iphi
      rechit[2] = 0.;            //iz
      // rechit[3] = energy; //energy
      rechit[3] = energy * hitsAndFractions[i].second;  //energy * fraction
                                                        // rechit[5] = hitsAndFractions[i].second; //fraction
    } else if (hitsAndFractions[i].first.subdetId() == EcalEndcap) {
      double energy = (*recHitsEE_->find(hitsAndFractions[i].first)).energy();
      EEDetId ee_id(hitsAndFractions[i].first);
      rechit[0] = ee_id.ix();     //ix
      rechit[1] = ee_id.iy();     //iy
      if (ee_id.zside() < 0)
        rechit[2] = -1.;  //iz
      if (ee_id.zside() > 0)
        rechit[2] = +1.;  //iz
      rechit[3] = energy * hitsAndFractions[i].second;  //energy * fraction
      // rechit[3] = energy; //energy
      // rechit[5] = hitsAndFractions[i].second; //fraction
    }else{
       edm::LogError("EcalClustersGraph") << "Rechit is not either EB or EE!!";
    }
    out[i] = rechit;
  }
  return out;
}

std::vector<double> EcalClustersGraph::computeVariables(const CaloCluster* seed, const CaloCluster* cluster) {
  std::vector<double> cl_vars(12);  //TODO == PUT dynamic configuration
  //showerShapes_ = computeShowerShapes(cluster,false);
  std::vector<int> clusterLocal = clusterPosition(cluster);

  cl_vars[0] = cluster->energy();                                //cl_energy
  cl_vars[1] = cluster->energy() / TMath::CosH(cluster->eta());  //cl_et
  cl_vars[2] = cluster->eta();                                   //cl_eta
  cl_vars[3] = cluster->phi();                                   //cl_phi
  cl_vars[4] = clusterLocal[0];                                  //cl_ieta/ix
  cl_vars[5] = clusterLocal[1];                                  //cl_iphi/iy
  cl_vars[6] = clusterLocal[2];                                  //cl_iz
  cl_vars[7] = deltaEta(seed->eta(), cluster->eta());            //cl_dEta
  cl_vars[8] = deltaPhi(seed->phi(), cluster->phi());            //cl_dPhi
  cl_vars[9] = seed->energy() - cluster->energy();               //cl_dEnergy
  cl_vars[10] =
      (seed->energy() / TMath::CosH(seed->eta())) - (cluster->energy() / TMath::CosH(cluster->eta()));  //cl_dEt
  cl_vars[11] = cluster->hitsAndFractions().size();                                                     // nxtals
  //   cl_vars[12] = showerShapes_[0]; //cl_r9
  //   cl_vars[13] = showerShapes_[1]; //cl_sigmaietaieta
  //   cl_vars[14] = showerShapes_[2]; //cl_sigmaietaiphi
  //   cl_vars[15] = showerShapes_[3]; //cl_sigmaiphiiphi
  //   cl_vars[16] = showerShapes_[4]; //cl_swiss_cross
  //   cl_vars[17] = showerShapes_[5]; //cl_nXtals
  //   cl_vars[18] = showerShapes_[6]; //cl_etaWidth
  //   cl_vars[19] = showerShapes_[7]; //cl_phiWidth
  return cl_vars;
}

std::vector<double> EcalClustersGraph::computeWindowVariables(const std::vector<std::vector<double>>& clusters) {
  size_t nCls = clusters.size();
  size_t nFeatures = clusters[0].size();
  std::vector<double> min(nFeatures);
  std::vector<double> max(nFeatures);
  std::vector<double> sum(nFeatures);
  for (const auto& vec : clusters) {
    for (size_t i = 0; i < nFeatures; i++) {
      const auto& x = vec[i];
      sum[i] += x;
      if (x < min[i])
        min[i] = x;
      if (x > max[i])
        max[i] = x;
    }
  }
  std::vector<double> out(18);
  out[0] = max[0];           // max_en_cluster
  out[1] = max[1];           // max_et_cluster
  out[2] = max[7];           // max_deta_cluster
  out[3] = max[8];           // max_dphi_cluster
  out[4] = max[9];           // max_den
  out[5] = max[10];          // max_det
  out[6] = min[0];           // min_en_cluster
  out[7] = min[1];           // min_et_cluster
  out[8] = min[7];           // min_deta
  out[9] = min[8];           // min_dphi
  out[10] = min[9];          // min_den
  out[11] = min[10];         // min_det
  out[12] = sum[0] / nCls;   // mean_en_cluster
  out[13] = sum[1] / nCls;   // mean_et_cluster
  out[14] = sum[7] / nCls;   // mean_deta
  out[15] = sum[8] / nCls;   // mean_dphi
  out[16] = sum[9] / nCls;   // mean_den
  out[17] = sum[10] / nCls;  // mean_det
  return out;
}

void EcalClustersGraph::fillVariables() {

  LogDebug("EcalClustersGraph") << "Preparing variables for all windows";
  
  //Looping on all the seeds (window)
  for (uint is = 0; is < nSeeds_; is++) {
    uint nClsInWindow = 0;
    const auto seedPointer = (*clusters_.at(is)).the_ptr().get();
    std::vector<std::vector<double>> unscaledClusterFeatures; 
    // Loop on all the clusters 
    for (uint ic = 0; ic < nCls_; ic++) {
      if (inWindows_.Get(is, ic) == 1) {
        const auto clPointer = (*clusters_.at(ic)).the_ptr().get();
        const auto & rawClX = computeVariables(seedPointer, clPointer);
        unscaledClusterFeatures.push_back(rawClX);
        inputs_.clustersX[is].push_back(SCProducerCache_->deepSCEvaluator->scaleClusterFeatures(rawClX));
        inputs_.hitsX[is].push_back(fillHits(clPointer));
        inputs_.isSeed[is].push_back(ic == is);
        nClsInWindow++;
      }
    }
    inputs_.windowX[is] = SCProducerCache_->deepSCEvaluator->scaleWindowFeatures(computeWindowVariables(unscaledClusterFeatures)); 
   
  }

  inputs_.batchSize = nSeeds_;

  LogDebug("EcalClustersGraph") << "N. Windows: "<< inputs_.clustersX.size();

  // LogDebug("EcalClustersGraph") << "Check hits:  Seed | Cluster | Hits";
  // for (uint i = 0; i< nSeeds_;i++){
  //   const size_t ncls = inputs_.hitsX[i].size();
  //   for (size_t j = 0; j<ncls; j++){
  //     const size_t nhits = inputs_.hitsX[i][j].size();
  //     std::cout << i << "|" << j << "/" << ncls  << " (isSeed:" << inputs_.isSeed[i][j] << ") | " << nhits  << std::endl;
  //     std::cout << "\t" ;
  //     for (size_t h=0; h<nhits; h++){
  //         std::cout  << inputs_.hitsX[i][j][h][3] << ", ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }
  LogDebug("EcalClustersGraph") << "End FillVariables";

}

void EcalClustersGraph::evaluateScores() {
  // Evaluate model
  const auto & scores = SCProducerCache_->deepSCEvaluator->evaluate(inputs_);
  for (uint i = 0; i < nSeeds_; ++i){
    uint k = 0;
    for (uint j = 0; j < nCls_; ++j) {
      if (inWindows_.Get(i, j) == 1){
        scoreMatrix_.Set(i, j, scores[i][k]);
        k++;
      }
      else{
          scoreMatrix_.Set(i, j, 0.);
      }
    }
  }
}

void EcalClustersGraph::printDebugInfo(){
  LogDebug("EcalClustersGraph") << "In window matrix:";
  for (uint i = 0; i < nSeeds_; ++i){
    for (uint j = 0; j < nCls_; ++j) {
      std::cout << inWindows_.Get(i, j) << ",";
    } 
    std::cout << std::endl;
  }
  LogDebug("EcalClustersGraph") << "Score matrix:";
  for (uint i = 0; i < nSeeds_; ++i){
    for (uint j = 0; j < nCls_; ++j) {
      std::cout << scoreMatrix_.Get(i, j) << ",";
    } 
    std::cout << std::endl;
  }
  LogDebug("EcalClustersGraph") << "Clusters ieta,iphi,iz,en";
  for (uint j = 0; j < nCls_; ++j) {
    const auto cluster = (*clusters_.at(j)).the_ptr().get();
    std::vector<int> clusterLocal = clusterPosition(cluster);
    std::cout << clusterLocal[0] << "," << clusterLocal[1]<< "," << clusterLocal[2] << "," << cluster->energy() << std::endl;
  } 
    
}

void EcalClustersGraph::setThresholds() {
  //test: place holder code
  thresholds_ = std::vector<float>(nSeeds_, 0.5);
}

void EcalClustersGraph::selectClusters() {
  //test
  clusterMatrix_ = scoreMatrix_.ReduceElements(1, 1, thresholds_, false);
  GraphMatrix clusterMatrixNoDuplicate_ = clusterMatrix_.RemoveDuplicates(1., false);
  for (size_type r = 0; r < clusterMatrixNoDuplicate_.nRows(); r++) {
    std::vector<float> row = clusterMatrixNoDuplicate_.GetRow(r);
    std::vector<float> subRow(row.begin(), row.begin() + clusterMatrixNoDuplicate_.nRows());
    if (GraphMatrix<float>().AllZeros(&subRow))
      clusterMatrix_.SetRowZero(r);
  }
  clusterMatrix_ = clusterMatrix_.RemoveDuplicates(1., false);
  //std::cout << "clusterMatrix: " << clusterMatrix_ << std::endl;
}

std::vector<std::pair<CalibratedClusterPtr, CalibratedClusterPtrVector>> EcalClustersGraph::getWindows() {
  std::vector<std::pair<CalibratedClusterPtr, CalibratedClusterPtrVector>> windows;
  for (size_type ir = 0; ir < clusterMatrix_.nRows(); ir++) {
    if (GraphMatrix<float>().AllZeros(clusterMatrix_.GetRow(ir)))
      continue;

    CalibratedClusterPtr seed = clusters_[ir];
    CalibratedClusterPtrVector clusters_inWindow;
    for (size_type ic = 0; ic < clusterMatrix_.nColumns(); ic++)
      if (clusterMatrix_.Get(ir, ic) != 0.)
        clusters_inWindow.push_back(clusters_[ic]);
    windows.push_back(std::make_pair(seed, clusters_inWindow));
  }
  return windows;
}
