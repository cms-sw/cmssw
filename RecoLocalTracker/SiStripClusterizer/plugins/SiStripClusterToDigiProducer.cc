#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "boost/foreach.hpp"
#include <numeric>

class SiStripClusterToDigiProducer : public edm::EDProducer  {

  typedef   edmNew::DetSetVector<SiStripCluster> ClusterCollection;
  typedef   edmNew::DetSet<SiStripCluster> DetClusterCollection;
  typedef   edmNew::DetSet<SiStripCluster>::const_iterator DetClusIter;


  typedef   edm::DetSetVector<SiStripDigi> DigiCollection;
  typedef   edm::DetSet<SiStripDigi> DetDigiCollection;
  typedef   edm::DetSet<SiStripDigi>::const_iterator DetDigiIter;


public:

  explicit SiStripClusterToDigiProducer(const edm::ParameterSet& conf);
  void produce(edm::Event&, const edm::EventSetup&);

private:

  void process(const ClusterCollection& input, std::vector<DetDigiCollection>& output_base);
  void initialize(const edm::EventSetup& es);
  void setDetId(const uint32_t id); 
  float gain(const uint16_t& strip)  const { return gainHandle->getStripGain( strip, gainRange ); }
  uint16_t applyGain(const uint16_t& strip,const uint16_t& adc );

  const edm::InputTag _inputTag;
  SiStripApvGain::Range gainRange;
  edm::ESHandle<SiStripGain> gainHandle;
  uint32_t gain_cache_id, detId;
  
};


SiStripClusterToDigiProducer::
SiStripClusterToDigiProducer(const edm::ParameterSet& conf) 
  : _inputTag( conf.getParameter<edm::InputTag>("ClusterProducer") ){
  
  produces< DigiCollection > ("ZeroSuppressed");
  produces< DigiCollection > ("VirginRaw"     );
  produces< DigiCollection > ("ProcessedRaw"  );
  produces< DigiCollection > ("ScopeMode"     );
  
}

void SiStripClusterToDigiProducer::
produce(edm::Event& event, const edm::EventSetup& es)  {
  
  initialize(es);
  
  std::vector<DetDigiCollection> output_base; 
  edm::Handle<ClusterCollection> input ;
  event.getByLabel(_inputTag,input);

  if(input.isValid())
    process(*input, output_base);


  std::auto_ptr< DigiCollection > outputZS(new DigiCollection(output_base) );
  std::auto_ptr< DigiCollection > outputVR(new DigiCollection() );
  std::auto_ptr< DigiCollection > outputPR(new DigiCollection() );
  std::auto_ptr< DigiCollection > outputSM(new DigiCollection() );

  event.put( outputZS, "ZeroSuppressed");
  event.put( outputVR, "VirginRaw"     );
  event.put( outputPR, "ProcessedRaw"  );
  event.put( outputSM, "ScopeMode"     );
}

void SiStripClusterToDigiProducer::
process(const ClusterCollection& input, std::vector<DetDigiCollection>& output_base)  {
 
  for(ClusterCollection::const_iterator it = input.begin(); it!=input.end(); ++it) {

    uint32_t detid=it->detId();

    setDetId(detid);
    DetDigiCollection detDigis(detid);

    DetClusIter clus(it->begin()), endclus(it->end());
    for(;clus!=endclus;clus++){
      size_t istrip     = 0;
      size_t width      = clus->amplitudes().size();
      size_t firstStrip = clus->firstStrip();
      uint16_t stripPos=firstStrip;
      for(;istrip<width;++istrip){
	detDigis.data.push_back( SiStripDigi( stripPos, applyGain(stripPos,clus->amplitudes()[istrip]) ) );
	stripPos++;
      }
    }
    
    if (detDigis.size()) 
      output_base.push_back(detDigis); 
  }
}

void SiStripClusterToDigiProducer::
initialize(const edm::EventSetup& es) {
  uint32_t g_cache_id = es.get<SiStripGainRcd>().cacheIdentifier();

  if(g_cache_id != gain_cache_id) {
    es.get<SiStripGainRcd>().get( gainHandle );
    gain_cache_id = g_cache_id;
  }
}


inline 
void SiStripClusterToDigiProducer::
setDetId(const uint32_t id) {
  gainRange =  gainHandle->getRange(id); 
  detId = id;
}

inline
uint16_t SiStripClusterToDigiProducer::
applyGain(const uint16_t& strip,const uint16_t& adc ) {

  if(adc > 255) throw cms::Exception("Invalid Charge") << " digi at strip " << strip << " has ADC out of range " << adc;
  if(adc > 253) return adc; //saturated, do not scale
  uint16_t charge = static_cast<uint16_t>( adc*gain(strip) + 0.5 ); //NB: here we revert the gain applied at the clusterizer level. for this reason the adc counts are multiplied by gain and not divided
  return ( charge > 1022 ? 255 : 
	  ( charge >  253 ? 254 : charge ));
}



#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripClusterToDigiProducer);
