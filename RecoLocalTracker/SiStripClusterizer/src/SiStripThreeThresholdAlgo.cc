#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripThreeThresholdAlgo.h"
#include <vector>
#include <algorithm>
#include <cmath>

// -----------------------------------------------------------------------------
//
SiStripThreeThresholdAlgo::SiStripThreeThresholdAlgo( const edm::ParameterSet& pset ) :

  SiStripClusterizerAlgo(pset),
  stripThr_(pset.getUntrackedParameter<double>("ChannelThreshold",2.)),
  seedThr_(pset.getUntrackedParameter<double>("SeedThreshold",3.)),
  clustThr_(pset.getUntrackedParameter<double>("ClusterThreshold",5.)),
  maxHoles_(pset.getUntrackedParameter<uint32_t>("MaxHolesInCluster",0)),
  charge_(0),
  seed_(false),
  sigmanoise2_(0.),
  strip_(0),
  first_(0),
  amps_(),
  digis_()

{
  LogTrace("UNDEFINED_CATEGORY")
    << "[SiStripThreeThresholdAlgo::" << __func__ << "]";
  
  amps_.reserve(768);
  digis_.reserve(768);
}

// -----------------------------------------------------------------------------
//
SiStripThreeThresholdAlgo::~SiStripThreeThresholdAlgo() {

  LogTrace("UNDEFINED_CATEGORY")
    << "[SiStripThreeThresholdAlgo::" << __func__ << "]";
}

// -----------------------------------------------------------------------------
//
void SiStripThreeThresholdAlgo::clusterize( const edm::DetSet<SiStripDigi>& digis, edm::DetSetVector<SiStripCluster>& clusters ) {

  LogTrace("UNDEFINED_CATEGORY")
    << "[SiStripThreeThresholdAlgo::" << __func__ << "]";

  edm::DetSet<SiStripCluster>& out = clusters.find_or_insert(digis.id);
  edm::DetSet<SiStripDigi>::const_iterator idigi = digis.begin();
  for (; idigi != digis.end(); idigi++) {
    add(out.data,out.id,idigi->strip(),idigi->adc());}
  endDet(out.data,out.id);
}

// -----------------------------------------------------------------------------
//
void SiStripThreeThresholdAlgo::add(std::vector<SiStripCluster>& data, const uint32_t& id, const uint16_t& istrip, const uint16_t& adc) {
 
  bool disable = quality()->IsStripBad(quality()->getRange(id),istrip);
  double stripnoise = noise()->getNoise(istrip,noise()->getRange(id));
  double stripgain = gain()->getStripGain(istrip,gain()->getRange(id));
  bool thresh = threshold(adc,stripnoise,disable);
  bool prox = proximity(istrip);

  //if strip dead and non zero record for later padding.
  if (adc && disable) digis_.push_back(istrip); 
  //if strip above threshold (and not dead) and within range of cluster, add strip.
  if (thresh && prox) {strip(istrip,adc,stripnoise,stripgain);}
  //if strip above threshold (and not dead) and beyond range of cluster, end cluster and add strip to new cluster.
  else if (thresh && !prox) {endCluster(data,id);strip(istrip,adc,stripnoise,stripgain);}
  //if strip below threshold (or dead) and beyond range of cluster, end cluster.
  else if (!thresh && !prox) {endCluster(data,id);}
  //if strip below threshold (or dead) and within range of cluster, continue. 
  //else {;} 

}

void SiStripThreeThresholdAlgo::endDet(std::vector<SiStripCluster>& data, const uint32_t& id) {
  endCluster(data,id); digis_.clear();
}

bool SiStripThreeThresholdAlgo::proximity(const uint16_t& istrip) const {
  if (amps_.empty()) return true;
  return (istrip <= maxHoles_+strip_+1); //loop to check they are dead?
}

bool SiStripThreeThresholdAlgo::threshold(const uint16_t& adc, const double& noise, const bool disable) const {  
  return (!disable && (adc >= static_cast<uint32_t>(noise * stripThr_)));
}

void SiStripThreeThresholdAlgo::pad(const uint16_t& left, const uint16_t& right) {
  for (uint16_t i=0;i<left;i++) {amps_.insert(amps_.begin(),0);first_--;}
  for (uint16_t k=0;k<right;k++) {amps_.push_back(0);}
}

void SiStripThreeThresholdAlgo::strip(const uint16_t& istrip, const uint16_t& adc, const double& noise, const double& gain) {


  if (adc >= noise*seedThr_) seed_ = true;
  if (amps_.empty()) first_ = istrip;
  else if (istrip - strip_ -1 > 0) {pad(0,istrip - strip_ -1);}

  //change by G. Bruno - 23/5/2008
  float stripCharge=(static_cast<float>(adc));
  if(stripCharge<254){
    stripCharge /= gain;	      
    if(stripCharge>511.5){stripCharge=255;}
    else if(stripCharge>253.5){stripCharge=254;}
  }  
  amps_.push_back(static_cast<uint16_t>(stripCharge+0.5));
  strip_ = istrip;
  //  charge_+=static_cast<float>(adc)/gain; 
  charge_+=stripCharge; 
  sigmanoise2_+=noise*noise/(gain*gain);
}

void SiStripThreeThresholdAlgo::endCluster(std::vector<SiStripCluster>& data, const uint32_t& id) {
  if (seed_ && (charge_ >= sqrt(sigmanoise2_) * clustThr_)) {
    if (find(digis_.begin(),digis_.end(),first_ -1) != digis_.end()) pad(1,0);
    if (find(digis_.begin(),digis_.end(),strip_+1) != digis_.end()) pad(0,1);
    data.push_back(SiStripCluster(id, first_, amps_.begin(),amps_.end()));
  }
  charge_ = 0.; sigmanoise2_ = 0.; seed_ = false; amps_.clear();
}
