#include "GenericSimClusterMapper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"

#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#ifdef PFLOW_DEBUG
#define LOGVERB(x) edm::LogVerbatim(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) edm::LogInfo(x)
#else
#define LOGVERB(x) LogTrace(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) LogDebug(x)
#endif

void GenericSimClusterMapper::
updateEvent(const edm::Event& ev) {
  ev.getByToken(_simClusterToken,_simClusterH);
}

void GenericSimClusterMapper::
buildClusters(const edm::Handle<reco::PFRecHitCollection>& input,
	      const std::vector<bool>& rechitMask,
	      const std::vector<bool>& seedable,
	      reco::PFClusterCollection& output) {
  const SimClusterCollection& simClusters = *_simClusterH;
  auto const& hits = *input;  
  
  // for quick indexing back to hit energy
  std::unordered_map<uint32_t, size_t> detIdToIndex(hits.size());  
  for( uint32_t i = 0; i < hits.size(); ++i ) {    
    detIdToIndex[hits[i].detId()] = i;
    auto ref = makeRefhit(input,i);    
  }
  
  for( const auto& sc : simClusters ) {
    output.emplace_back();
    reco::PFCluster& back = output.back();
    edm::Ref<std::vector<reco::PFRecHit> > seed;    
    double energy = 0.0, highest_energy = 0.0;
    auto hitsAndFractions = sc.hits_and_fractions() ;
    for( const auto& hAndF : hitsAndFractions ) {
      auto itr = detIdToIndex.find(hAndF.first);
      if( itr == detIdToIndex.end() ) continue; // hit wasn't saved in reco
      auto ref = makeRefhit(input,itr->second);            
      const double hit_energy = hAndF.second * ref->energy();
      energy += hit_energy;  
      back.addRecHitFraction(reco::PFRecHitFraction(ref, hAndF.second));
      if( hit_energy > highest_energy || highest_energy == 0.0) {
	highest_energy = hit_energy;
	seed = ref;
      }
    }    
    if( back.hitsAndFractions().size() != 0 ) {
      back.setSeed(seed->detId());
      back.setEnergy(energy);   
      back.setCorrectedEnergy(energy);
    } else {
      back.setSeed(-1);
      back.setEnergy(0.f);
    }
  }
}

