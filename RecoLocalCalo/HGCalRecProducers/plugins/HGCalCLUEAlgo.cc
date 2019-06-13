#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalCLUEAlgo.h"

// Geometry
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
//
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "tbb/task_arena.h"
#include "tbb/tbb.h"
#include <limits>

using namespace hgcal_clustering;


void HGCalCLUEAlgo::populate(const HGCRecHitCollection &hits) {
  // loop over all hits and create the Hexel structure, skip energies below ecut

  if (dependSensor_) {
    // for each layer and wafer calculate the thresholds (sigmaNoise and energy)
    // once
    computeThreshold();
  }
  
  for (unsigned int i = 0; i < hits.size(); ++i) {
    const HGCRecHit &hgrh = hits[i];
    DetId detid = hgrh.detid();
    unsigned int layerOnSide = (rhtools_.getLayerWithOffset(detid) - 1);

    // set sigmaNoise default value 1 to use kappa value directly in case of
    // sensor-independent thresholds
    float sigmaNoise = 1.f;
    if (dependSensor_) {
      int thickness_index = rhtools_.getSiThickIndex(detid);
      if (thickness_index == -1) thickness_index = 3;
      double storedThreshold = thresholds_[layerOnSide][thickness_index];
      sigmaNoise = v_sigmaNoise_[layerOnSide][thickness_index];

      if (hgrh.energy() < storedThreshold)
        continue;  // this sets the ZS threshold at ecut times the sigma noise
                   // for the sensor
    }
    if (!dependSensor_ && hgrh.energy() < ecut_) continue;
    const GlobalPoint position(rhtools_.getPosition(detid));

    int offset = ((rhtools_.zside(detid) + 1) >> 1)*maxlayer;
    int layer = layerOnSide + offset;
    cells_[layer].detid.emplace_back(detid);
    cells_[layer].isSilic.emplace_back(rhtools_.isSilicon(detid));
    cells_[layer].x.emplace_back(position.x());
    cells_[layer].y.emplace_back(position.y());
    cells_[layer].eta.emplace_back(position.eta());
    cells_[layer].phi.emplace_back(position.phi());
    cells_[layer].weight.emplace_back(hgrh.energy());
    cells_[layer].sigmaNoise.emplace_back(sigmaNoise);
  }
}

void HGCalCLUEAlgo::prepareDataStructures(unsigned int l)
{
  auto cellsSize = cells_[l].detid.size();
  cells_[l].rho.resize(cellsSize,0);
  cells_[l].delta.resize(cellsSize,9999999);
  cells_[l].nearestHigher.resize(cellsSize,-1);
  cells_[l].clusterIndex.resize(cellsSize,-1);
  cells_[l].followers.resize(cellsSize);
  cells_[l].isSeed.resize(cellsSize,false);
  
}

// Create a vector of Hexels associated to one cluster from a collection of
// HGCalRecHits - this can be used directly to make the final cluster list -
// this method can be invoked multiple times for the same event with different
// input (reset should be called between events)
void HGCalCLUEAlgo::makeClusters() {
  // assign all hits in each layer to a cluster core
  tbb::this_task_arena::isolate([&] {
    tbb::parallel_for(size_t(0), size_t(2 * maxlayer + 2), [&](size_t i) {
      prepareDataStructures(i);
      //std::cout << "---Starting process layer " << i << std::endl;
      //if (i!=58) return;
      HGCalLayerTiles lt;
      lt.clear();
      lt.fill(cells_[i].x,cells_[i].y, cells_[i].eta, cells_[i].phi, cells_[i].isSilic);
      
      float delta_c;  // maximum search distance (critical distance) for local
                  // density calculation
      if (i%maxlayer < lastLayerEE)
        delta_c = vecDeltas_[0];
      else if (i%maxlayer < lastLayerFH)
        delta_c = vecDeltas_[1];
      else
        delta_c = vecDeltas_[2];

      //float delta_r = 0.0315;
      
      //prepareDataStructures(i);
      //if (i!=94) return;
      calculateLocalDensity(lt, i, delta_c);
      calculateDistanceToHigher(lt, i, delta_c);
      numberOfClustersPerLayer_[i] = findAndAssignClusters(i,delta_c);
      //std::cout << "makeClusters: " << i << " | " << numberOfClustersPerLayer_[i] << " | " << cells_[i].x.size() << std::endl;
    });
  });
  //Now that we have the density per point we can store it
  for(unsigned int i=0; i< 2 * maxlayer + 2; ++i) {
    //std::cout << i << std::endl; //}
    setDensity(i); }
  //std::cout << "end makeClusters" << std::endl;
}

std::vector<reco::BasicCluster> HGCalCLUEAlgo::getClusters(bool) {
  //std::cout << "debug5" << std::endl;
  std::vector<int> offsets(numberOfClustersPerLayer_.size(),0);

  int maxClustersOnLayer = numberOfClustersPerLayer_[0];

  for(unsigned layerId = 1; layerId<offsets.size(); ++layerId)
  {
    offsets[layerId] = offsets[layerId-1] +numberOfClustersPerLayer_[layerId-1];

    maxClustersOnLayer = std::max(maxClustersOnLayer, numberOfClustersPerLayer_[layerId]);
  }

  //std::cout << "debug6" << std::endl;
  auto totalNumberOfClusters = offsets.back()+numberOfClustersPerLayer_.back();
  clusters_v_.resize(totalNumberOfClusters);
  std::vector<std::vector<int> > cellsIdInCluster;
  cellsIdInCluster.reserve(maxClustersOnLayer);

  for(unsigned int layerId = 0; layerId < 2 * maxlayer + 2; ++layerId)
  {
    cellsIdInCluster.resize(numberOfClustersPerLayer_[layerId]);
    auto& cellsOnLayer = cells_[layerId];
    unsigned int numberOfCells = cellsOnLayer.detid.size();
    auto firstClusterIdx = offsets[layerId];
    
    for (unsigned int i = 0; i < numberOfCells; ++i )
    {   
      auto clusterIndex = cellsOnLayer.clusterIndex[i];
      if(clusterIndex != -1)
        cellsIdInCluster[clusterIndex].push_back(i);
    }
    

    std::vector<std::pair<DetId, float>> thisCluster;
    //std::cout << "debug7" << std::endl;
    for(auto& cl: cellsIdInCluster)
    {
      auto position = calculatePosition(cl, layerId);
      float energy = 0.f;
      int seedDetId = -1;
      
      for(auto cellIdx : cl)
      {
        energy+= cellsOnLayer.weight[cellIdx];
        thisCluster.emplace_back(cellsOnLayer.detid[cellIdx],1.f);
        if(cellsOnLayer.isSeed[cellIdx])
        {
          seedDetId = cellsOnLayer.detid[cellIdx];
        }
      }
      auto globalClusterIndex = cellsOnLayer.clusterIndex[cl[0]] +  firstClusterIdx;
      
      clusters_v_[globalClusterIndex]=reco::BasicCluster(energy, position, reco::CaloID::DET_HGCAL_ENDCAP, thisCluster, algoId_);
      clusters_v_[globalClusterIndex].setSeed(seedDetId);
      thisCluster.clear();
    }

    cellsIdInCluster.clear();

  }
  //std::cout << "debug8" << std::endl;
  return clusters_v_;

}



math::XYZPoint HGCalCLUEAlgo::calculatePosition(const std::vector<int> &v, const unsigned int layerId) const {
  
  float total_weight = 0.f;
  float x = 0.f;
  float y = 0.f;

  unsigned int maxEnergyIndex = 0;
  float maxEnergyValue = 0.f;
  
  auto& cellsOnLayer = cells_[layerId];


  // loop over hits in cluster candidate
  // determining the maximum energy hit
  for (auto i : v) {
    total_weight += cellsOnLayer.weight[i];
    if (cellsOnLayer.weight[i] > maxEnergyValue) {
      maxEnergyValue = cellsOnLayer.weight[i];
      maxEnergyIndex = i;
    }
  }

  // Si cell or Scintillator. Used to set approach and parameters
  auto thick = rhtools_.getSiThickIndex(cellsOnLayer.detid[maxEnergyIndex]);
  bool isSiliconCell = (thick != -1);


// TODO: this is recomputing everything twice and overwriting the position with log weighting position
  if(isSiliconCell)
  {
    float total_weight_log = 0.f;
    float x_log = 0.f;
    float y_log = 0.f;
    for (auto i : v) {
      float rhEnergy = cellsOnLayer.weight[i];
      float Wi = std::max(thresholdW0_[thick] + std::log(rhEnergy / total_weight), 0.);
      x_log += cellsOnLayer.x[i] * Wi;
      y_log += cellsOnLayer.y[i] * Wi;
      total_weight_log += Wi;
    }

    total_weight = total_weight_log;
    x = x_log;
    y = y_log;
  }
  else
  {
    for (auto i : v) {

      float rhEnergy = cellsOnLayer.weight[i];

      x += cellsOnLayer.x[i] * rhEnergy;
      y += cellsOnLayer.y[i] * rhEnergy;
      
    }
  }
  if (total_weight != 0.) {
    float inv_tot_weight = 1.f / total_weight;
    return math::XYZPoint(x * inv_tot_weight, y * inv_tot_weight, rhtools_.getPosition(cellsOnLayer.detid[maxEnergyIndex]).z());
  }
  else return math::XYZPoint(0.f, 0.f, 0.f);
}



void HGCalCLUEAlgo::calculateLocalDensity(const HGCalLayerTiles& lt, const unsigned int layerId, float delta_c)  
{
  auto& cellsOnLayer = cells_[layerId];
  unsigned int numberOfCells = cellsOnLayer.detid.size();
  for(unsigned int i = 0; i < numberOfCells; i++) {
    //std::cout << "CELL: " << i << "\n";
    float delta_c_(0);
    if (cellsOnLayer.isSilic[i]) {
      delta_c_=delta_c;
      //std::cout << "delta for silicon: " << delta_c_ << std::endl;
      //std::cout << "x-y: " << cellsOnLayer.x[i] << "|" << cellsOnLayer.y[i] << "|" << lt.getXBin(cellsOnLayer.x[i]) << "|" << lt.getYBin(cellsOnLayer.y[i]) << std::endl;
      //std::cout << "eta-phi: " << cellsOnLayer.eta[i] << "|" << cellsOnLayer.phi[i] << "|" << lt.getEtaBin(cellsOnLayer.eta[i]) << "|" << lt.getPhiBin(cellsOnLayer.phi[i]) << std::endl;
      std::array<int,4> search_box = lt.searchBox(cellsOnLayer.x[i] - delta_c_, cellsOnLayer.x[i] + delta_c_, cellsOnLayer.y[i] - delta_c_, cellsOnLayer.y[i] + delta_c_);
    
      for(int xBin = search_box[0]; xBin < search_box[1]+1; ++xBin) {
	for(int yBin = search_box[2]; yBin < search_box[3]+1; ++yBin) {
        
	  int binId = lt.getGlobalBinByBin(xBin,yBin);
	  size_t binSize = lt[binId].size();
        
	  for (unsigned int j = 0; j < binSize; j++) {
	    unsigned int otherId = lt[binId][j];
	    //std::cout << otherId << std::endl;
	    //std::cout << "x-y in search bin:" << xBin << "|" << yBin << "|" << binId << "|" << lt[binId].size() << std::endl;
	    if (!cellsOnLayer.isSilic[otherId]) continue;
	    if(distance(i, otherId, layerId, false) < delta_c_) {
	      cellsOnLayer.rho[i] += (i == otherId ? 1.f : 0.5f) * cellsOnLayer.weight[otherId];
	    }
	  }
	}
      }
      //std::cout << "calculate local density: " << i << " rho: " << cellsOnLayer.rho[i] << std::endl;
      //std::cout << "calculate local density: " << i << " delta: " << cellsOnLayer.delta[i] << std::endl;
    } else {
      delta_c_ = 0.0315;
      //std::cout << "delta for scintillator: " << delta_c_ << std::endl;
      //std::cout << "eta-phi: " << cellsOnLayer.eta[i] << "|" << cellsOnLayer.phi[i] << "|" << lt.getEtaBin(cellsOnLayer.eta[i]) << "|" << lt.getPhiBin(cellsOnLayer.phi[i]) << std::endl;
      std::array<int,4> search_box = lt.searchBoxEtaPhi(cellsOnLayer.eta[i] - delta_c_, cellsOnLayer.eta[i] + delta_c_, cellsOnLayer.phi[i] - delta_c_, cellsOnLayer.phi[i] + delta_c_);
      //std::cout << "search box: " << search_box[0] << "|" << search_box[1] << "|" << search_box[2] << "|" << search_box[3] << std::endl;
      cellsOnLayer.rho[i] += cellsOnLayer.weight[i];
	
      float northeast(0), northwest(0), southeast(0), southwest(0);
      
      for(int etaBin = search_box[0]; etaBin < search_box[1]+1; ++etaBin) {
	for(int phiBin = search_box[2]; phiBin < search_box[3]+1; ++phiBin) {
	  
	  int binId = lt.getGlobalBinByBinEtaPhi(etaBin,phiBin);
	  //std::cout << "eta-phi in search bin:" << etaBin << "|" << phiBin << "|" << binId << "|" << lt[binId].size() << std::endl;
	  size_t binSize = lt[binId].size();
	  for (unsigned int j = 0; j < binSize; j++) {
	    unsigned int otherId = lt[binId][j];
	    //std::cout << "otherId: " << otherId << " binId: " << binId << " i: "<< i << std::endl;
	    if (cellsOnLayer.isSilic[otherId]) continue;
	    if(distance(i, otherId, layerId, true) < delta_c_) {
	      float otherPhi=cellsOnLayer.phi[otherId];
	      float iPhi=cellsOnLayer.phi[i];
	      otherPhi += (otherPhi*iPhi >= 0 || abs(iPhi) < 1.) ? 0. : iPhi > 0. ? 2*M_PI : -2*M_PI;
	      float otherEta=cellsOnLayer.eta[otherId];
	      float iEta=cellsOnLayer.eta[i];
	      if ((otherEta > iEta && otherPhi > iPhi) || (otherEta > iEta && otherPhi == iPhi) || (otherEta == iEta && otherPhi > iPhi)) northwest+=0.5*cellsOnLayer.weight[otherId];
	      if ((otherEta > iEta && otherPhi < iPhi) || (otherEta > iEta && otherPhi == iPhi) || (otherEta == iEta && otherPhi < iPhi)) northeast+=0.5*cellsOnLayer.weight[otherId];
	      if ((otherEta < iEta && otherPhi > iPhi) || (otherEta < iEta && otherPhi == iPhi) || (otherEta == iEta && otherPhi > iPhi)) southwest+=0.5*cellsOnLayer.weight[otherId];
	      if ((otherEta < iEta && otherPhi < iPhi) || (otherEta == iEta && otherPhi < iPhi) || (otherEta < iEta && otherPhi == iPhi)) southeast+=0.5*cellsOnLayer.weight[otherId];
	    }
	  }
	}
      }
      float neighborsval = (std::max(northeast, northwest) > std::max(southeast, southwest)) ? std::max(northeast, northwest) : std::max(southeast, southwest);
      cellsOnLayer.rho[i] += neighborsval;
//      std::cout << " Cell: " << i
//		<< " eta: " <<  cellsOnLayer.eta[i]
//		<< " phi: " << cellsOnLayer.phi[i]
//		<< " energy: " << cellsOnLayer.weight[i]
//		<< " density: " << cellsOnLayer.rho[i]
//		<< "\n";
    }
  }
}


void HGCalCLUEAlgo::calculateDistanceToHigher(const HGCalLayerTiles& lt, const unsigned int layerId, float delta_c) {

  auto& cellsOnLayer = cells_[layerId];
  unsigned int numberOfCells = cellsOnLayer.detid.size();

  for(unsigned int i = 0; i < numberOfCells; i++) {
    // initialize delta and nearest higher for i
    float maxDelta = std::numeric_limits<float>::max();
    float i_delta = maxDelta;
    int i_nearestHigher = -1;
    float delta_c_ = 0;
    if (cellsOnLayer.isSilic[i]) {
      delta_c_=delta_c;
      // get search box for ith hit
      // guarantee to cover a range "outlierDeltaFactor_*delta_c"
      auto range = outlierDeltaFactor_*delta_c_;
      std::array<int,4> search_box = lt.searchBox(cellsOnLayer.x[i]  - range, cellsOnLayer.x[i] + range, cellsOnLayer.y[i] - range, cellsOnLayer.y[i] + range);
      // loop over all bins in the search box
      for(int xBin = search_box[0]; xBin < search_box[1]+1; ++xBin) {
	for(int yBin = search_box[2]; yBin < search_box[3]+1; ++yBin) {
        
	  // get the id of this bin
	  size_t binId = lt.getGlobalBinByBin(xBin,yBin);
	  // get the size of this bin
	  size_t binSize = lt[binId].size();

	  // loop over all hits in this bin
	  for (unsigned int j = 0; j < binSize; j++) {
	    unsigned int otherId = lt[binId][j];
	    if (!cellsOnLayer.isSilic[otherId]) continue;
	    float dist = distance(i, otherId, layerId, false);
	    bool foundHigher = cellsOnLayer.rho[otherId] > cellsOnLayer.rho[i];


          // if dist == i_delta, then last comer being the nearest higher
	    if(foundHigher && dist <= i_delta) {
	      // update i_delta
	      i_delta = dist;
	      // update i_nearestHigher
	      i_nearestHigher = otherId;            
	    }
	  }
	}
      }

      bool foundNearestHigherInSearchBox = (i_delta != maxDelta);
      if (foundNearestHigherInSearchBox){
	cellsOnLayer.delta[i] = i_delta;
	cellsOnLayer.nearestHigher[i] = i_nearestHigher;
      } else {
	// otherwise delta is guaranteed to be larger outlierDeltaFactor_*delta_c
	// we can safely maximize delta to be maxDelta
	cellsOnLayer.delta[i] = maxDelta;
	cellsOnLayer.nearestHigher[i] = -1;
      }
      //std::cout << "calculate distance: " << i << " rho: " << cellsOnLayer.rho[i] << std::endl;
      //std::cout << "calculate distance: " << i << " delta: " << cellsOnLayer.delta[i] << std::endl;
    } else {

      //std::cout << "calculateDistanceToHigher, cell: " << i << std::endl;
      // get search box for ith hit
      // guarantee to cover a range "outlierDeltaFactor_*delta_c"

      delta_c_ = 0.0315;
      auto range = outlierDeltaFactor_*delta_c_;
      std::array<int,4> search_box = lt.searchBoxEtaPhi(cellsOnLayer.eta[i]  - range, cellsOnLayer.eta[i] + range, cellsOnLayer.phi[i] - range, cellsOnLayer.phi[i] + range);
      //std::cout << "search box: " << search_box[0] << "|" << search_box[1] << "|" << search_box[2] << "|" << search_box[3] << std::endl;
      // loop over all bins in the search box
      for(int xBin = search_box[0]; xBin < search_box[1]+1; ++xBin) {
	for(int yBin = search_box[2]; yBin < search_box[3]+1; ++yBin) {
        
	  // get the id of this bin
	  //std::cout << "debug: " << xBin << "|" << yBin << std::endl;
	  size_t binId = lt.getGlobalBinByBinEtaPhi(xBin,yBin);
	  // get the size of this bin
	  size_t binSize = lt[binId].size();
	  //std::cout << "eta-phi in search bin:" << xBin << "|" << yBin << "|" << binId << "|" << lt[binId].size() << std::endl;
	  // loop over all hits in this bin
	  for (unsigned int j = 0; j < binSize; j++) {
	    unsigned int otherId = lt[binId][j];
	    if (cellsOnLayer.isSilic[otherId]) continue;
	    //std::cout << "otherId: " << otherId << " binId: " << binId << " i: "<< i << std::endl;
	    float dist = distance(i, otherId, layerId, true);
	    bool foundHigher = cellsOnLayer.rho[otherId] > cellsOnLayer.rho[i];

          // if dist == i_delta, then last comer being the nearest higher
	    if(foundHigher && dist <= i_delta) {
	      // update i_delta
	      i_delta = dist;
	      // update i_nearestHigher
	      i_nearestHigher = otherId;
	    }
	  }
	}
      }

      bool foundNearestHigherInSearchBox = (i_delta != maxDelta);
      if (foundNearestHigherInSearchBox){
	cellsOnLayer.delta[i] = i_delta;
	cellsOnLayer.nearestHigher[i] = i_nearestHigher;
      } else {
	// otherwise delta is guaranteed to be larger outlierDeltaFactor_*delta_c
	// we can safely maximize delta to be maxDelta
	cellsOnLayer.delta[i] = maxDelta;
	cellsOnLayer.nearestHigher[i] = -1;
      }
    }
  }
}


int HGCalCLUEAlgo::findAndAssignClusters(const unsigned int layerId, float delta_c ) {
   
  // this is called once per layer and endcap...
  // so when filling the cluster temporary vector of Hexels we resize each time
  // by the number  of clusters found. This is always equal to the number of
  // cluster centers...
  unsigned int nClustersOnLayer = 0;
  auto& cellsOnLayer = cells_[layerId];
  unsigned int numberOfCells = cellsOnLayer.detid.size();
  std::vector<int> localStack;
  // find cluster seeds and outlier  
  for(unsigned int i = 0; i < numberOfCells; i++) {
    float rho_c = kappa_ * cellsOnLayer.sigmaNoise[i];
    float delta_c_ = 0;
    if (!cellsOnLayer.isSilic[i]) {
      delta_c_ = 0.0315;
    } else {
      delta_c_ = delta_c;
    }

    // initialize clusterIndex
    cellsOnLayer.clusterIndex[i] = -1;
    bool isSeed = (cellsOnLayer.delta[i] > delta_c_) && (cellsOnLayer.rho[i] >= rho_c);
    bool isOutlier = (cellsOnLayer.delta[i] > outlierDeltaFactor_*delta_c_) && (cellsOnLayer.rho[i] < rho_c);
    if (isSeed) 
    {
      cellsOnLayer.clusterIndex[i] = nClustersOnLayer;
      cellsOnLayer.isSeed[i] = true;
      nClustersOnLayer++;
      localStack.push_back(i);
    
    } else if (!isOutlier) {
      cellsOnLayer.followers[cellsOnLayer.nearestHigher[i]].push_back(i);   
    }
    //std::cout << "assign clusters: " << i << " rho: " << cellsOnLayer.rho[i] << std::endl;
    //std::cout << "assign clusters: " << i << " delta: " << cellsOnLayer.delta[i] << std::endl;
  }
  
  // need to pass clusterIndex to their followers
  while (!localStack.empty()) {
    int endStack = localStack.back();
    auto& thisSeed = cellsOnLayer.followers[endStack];
    localStack.pop_back();

    // loop over followers
    for( int j : thisSeed){
      // pass id to a follower
      cellsOnLayer.clusterIndex[j] = cellsOnLayer.clusterIndex[endStack];
      // push this follower to localStack
      localStack.push_back(j);
    }
  }
  //std::cout << "nClustersOnLayer: " << nClustersOnLayer << std::endl; 
  return nClustersOnLayer;
}

void HGCalCLUEAlgo::computeThreshold() {
  // To support the TDR geometry and also the post-TDR one (v9 onwards), we
  // need to change the logic of the vectors containing signal to noise and
  // thresholds. The first 3 indices will keep on addressing the different
  // thicknesses of the Silicon detectors, while the last one, number 3 (the
  // fourth) will address the Scintillators. This change will support both
  // geometries at the same time.

  if (initialized_) return;  // only need to calculate thresholds once

  initialized_ = true;

  std::vector<double> dummy;
  const unsigned maxNumberOfThickIndices = 3;
  dummy.resize(maxNumberOfThickIndices + 1, 0);  // +1 to accomodate for the Scintillators
  thresholds_.resize(maxlayer, dummy);
  v_sigmaNoise_.resize(maxlayer, dummy);

  for (unsigned ilayer = 1; ilayer <= maxlayer; ++ilayer) {
    for (unsigned ithick = 0; ithick < maxNumberOfThickIndices; ++ithick) {
      float sigmaNoise = 0.001f * fcPerEle_ * nonAgedNoises_[ithick] * dEdXweights_[ilayer] /
                         (fcPerMip_[ithick] * thicknessCorrection_[ithick]);
      thresholds_[ilayer - 1][ithick] = sigmaNoise * ecut_;
      v_sigmaNoise_[ilayer - 1][ithick] = sigmaNoise;
    }
    float scintillators_sigmaNoise = 0.001f * noiseMip_ * dEdXweights_[ilayer];
    thresholds_[ilayer - 1][maxNumberOfThickIndices] = ecut_ * scintillators_sigmaNoise;
    v_sigmaNoise_[ilayer - 1][maxNumberOfThickIndices] = scintillators_sigmaNoise;
  }
}

void HGCalCLUEAlgo::setDensity(const unsigned int layerId){

  auto& cellsOnLayer = cells_[layerId];
  unsigned int numberOfCells = cellsOnLayer.detid.size();
  //std::cout << numberOfCells << std::endl;
  for (unsigned int i = 0; i< numberOfCells; ++i) {
    //std::cout << i << " density: " << cellsOnLayer.rho[i] << std::endl;
    density_[ cellsOnLayer.detid[i] ] =   cellsOnLayer.rho[i] ;
  }
}

Density HGCalCLUEAlgo::getDensity() {
  return density_;
}
