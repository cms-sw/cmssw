#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// #define VIDEBUG

#ifdef VIDEBUG
#define COUT std::cout
#else
#define COUT LogDebug("StripClusterizerAlgorithm")
#endif

#include <string>
#include <algorithm>
#include <cassert>

void StripClusterizerAlgorithm::
initialize(const edm::EventSetup& es) {
  uint32_t n_cache_id = es.get<SiStripNoisesRcd>().cacheIdentifier();
  uint32_t g_cache_id = es.get<SiStripGainRcd>().cacheIdentifier();
  uint32_t q_cache_id = es.get<SiStripQualityRcd>().cacheIdentifier();

  bool mod=false;
  if(g_cache_id != gain_cache_id) {
    es.get<SiStripGainRcd>().get( gainHandle );
    gain_cache_id = g_cache_id;
    mod=true;
  }
  if(n_cache_id != noise_cache_id) {
    es.get<SiStripNoisesRcd>().get( noiseHandle );
    noise_cache_id = n_cache_id;
    mod=true;
  }
  if(q_cache_id != quality_cache_id) {
    es.get<SiStripQualityRcd>().get( qualityLabel, qualityHandle );
    quality_cache_id = q_cache_id;
    mod=true;
  }

  if (mod) { 
    // redo indexing!
    theCabling = qualityHandle->cabling();
    assert(theCabling); 
    auto const & conn = cabling()->connected();
    COUT << "cabling " << conn.size() << std::endl;
    detIds.clear();
    detIds.reserve(conn.size());
    for (auto const & c : conn) { if (!isModuleBad(c.first)) detIds.push_back(c.first);}
    indices.clear();
    indices.resize(detIds.size());
    COUT << "good detIds " << detIds.size() << std::endl;

    if (0==detIds.size()) {
       ind=0; detId=0; return;
    }

    {
      connections.clear();
      connections.resize(detIds.size());
      //connections (slow, not a big deal)
      auto const & conns = cabling()->getDetCabling();
      for (auto i=0U; i<detIds.size(); ++i) {
	auto c = conns.find(detIds[i]);
	if (c!=conns.end()) connections[i]=(*c).second;
      }
      

    }


    { // quality
      std::vector<uint32_t> dum; qualityHandle->getDetIds(dum); 
      assert(dum.size()<invalidI);
      unsigned short j=0, i=0;
      while (i<dum.size() && j<detIds.size()) {
	if (dum[i]<detIds[j]) ++i;
	else if (detIds[j]<dum[i]) {indices[j].qi=invalidI; ++j;}
	else {
	  indices[j].qi=i; ++i; ++j;
	}
      }
      unsigned int  nn=0;
      for(auto k=0U; k<detIds.size();++k) { if (indices[k].qi<invalidI) {++nn; assert(dum[indices[k].qi]==detIds[k]);}}
      assert(nn<=dum.size());
      COUT << "quality " << dum.size() << " " <<nn<< std::endl;
    }
    { //noise
      std::vector<uint32_t> dum; noiseHandle->getDetIds(dum); 
      assert(dum.size()<invalidI);
      unsigned short j=0, i=0;
      while (i<dum.size() && j<detIds.size()) {
	if (dum[i]<detIds[j]) ++i;
	else if (detIds[j]<dum[i]) {indices[j].ni=invalidI; ++j;}
	else {
	  indices[j].ni=i; ++i; ++j;
	}
      }
      unsigned int  nn=0;
      for(auto k=0U; k<detIds.size();++k) { if (indices[k].ni<invalidI) {++nn; assert(dum[indices[k].ni]==detIds[k]);}}
      assert(nn<=dum.size());
      COUT << "noise " << dum.size() << " " <<nn<< std::endl;
    }
    { //gain
      std::vector<uint32_t> dum; gainHandle->getDetIds(dum); 
      assert(dum.size()<invalidI);
      unsigned short j=0, i=0;
      while (i<dum.size() && j<detIds.size()) {
	if (dum[i]<detIds[j]) ++i;
	else if (detIds[j]<dum[i]) {indices[j].gi=invalidI; ++j;}
	else {
	  indices[j].gi=i; ++i; ++j;
	}
      }
      unsigned int  nn=0;
      for(auto k=0U; k<detIds.size();++k) { if (indices[k].gi<invalidI) {++nn; assert(dum[indices[k].gi]==detIds[k]);}}
      assert(nn<=dum.size());
      COUT << "gain " << dum.size() << " " <<nn<< std::endl;
    }
  }

  if (0==detIds.size()) {
       ind=0; detId=0; return;
  }

  // initalize first det
  ind = 0;
  detId = detIds[ind];
  
  noiseRange = noiseHandle->getRangeByPos(indices[ind].ni);
  gainRange = gainHandle->getRangeByPos(indices[ind].gi);
  qualityRange = qualityHandle->getRangeByPos(indices[ind].qi);
  
}
  

bool StripClusterizerAlgorithm::
setDetId(const uint32_t id) {
  if (id==detId) return true; // rare....
  /*
  // priority to sequential scan
  if (id==detIds[ind+1]) {
    ++ind;
  } else if unlikely( (id>detId) &  (id<detIds[ind+1]) ) {
//      edm::LogWarning("StripClusterizerAlgorithm") 
//	<<"id " << id << " not connected. this is impossible on data "
//	<< "old id " << detId << std::endl;
      return false; 
    }   else{
  */
    auto b = detIds.begin();
    auto e = detIds.end();
    // if (id>detId) b+=ind;
    // else e=b+ind;
    auto p = std::lower_bound(b,e,id);
    if (p==e || id!=(*p)) {
#ifdef NOT_ON_MONTECARLO
      edm::LogWarning("StripClusterizerAlgorithm") 
	<<"id " << id << " not connected. this is impossible on data "
	<< "old id " << detId << std::endl;
#endif
      return false;
    }
    ind = p-detIds.begin();
//  }

  detId = id;
  noiseRange = noiseHandle->getRangeByPos(indices[ind].ni);
  gainRange = gainHandle->getRangeByPos(indices[ind].gi);
  qualityRange = qualityHandle->getRangeByPos(indices[ind].qi);

#ifdef EDM_ML_DEBUG
  assert(detIds[ind]==detId); 
  auto oldg =  gainHandle->getRange(id);
  assert(oldg==gainRange);
  auto oldn = noiseHandle->getRange(id);
  assert(oldn==noiseRange);
  auto oldq = qualityHandle->getRange(id);
  assert(oldq==qualityRange);
#endif

  return true;
}

void StripClusterizerAlgorithm::clusterize(const   edm::DetSetVector<SiStripDigi>& input,  output_t& output) {clusterize_(input, output);}
void StripClusterizerAlgorithm::clusterize(const edmNew::DetSetVector<SiStripDigi>& input, output_t& output) {clusterize_(input, output);}

StripClusterizerAlgorithm::
InvalidChargeException::InvalidChargeException(const SiStripDigi& digi)
  : cms::Exception("Invalid Charge") {
  std::stringstream s;
  s << "Digi charge of " << digi.adc() << " ADC "
    << "is out of range on strip " << digi.strip() << ".  ";
  this->append(s.str());
}
