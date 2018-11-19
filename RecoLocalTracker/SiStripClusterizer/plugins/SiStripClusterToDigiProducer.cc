#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
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
#include <numeric>

class SiStripClusterToDigiProducer : public edm::stream::EDProducer<>  {

  typedef   edmNew::DetSetVector<SiStripCluster> ClusterCollection;
  typedef   edmNew::DetSet<SiStripCluster> DetClusterCollection;
  typedef   edmNew::DetSet<SiStripCluster>::const_iterator DetClusIter;


  typedef   edm::DetSetVector<SiStripDigi> DigiCollection;
  typedef   edm::DetSet<SiStripDigi> DetDigiCollection;
  typedef   edm::DetSet<SiStripDigi>::const_iterator DetDigiIter;


public:

  explicit SiStripClusterToDigiProducer(const edm::ParameterSet& conf);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:

  void process(const ClusterCollection& input, std::vector<DetDigiCollection>& output_base);
  void initialize(const edm::EventSetup& es);
  void setDetId(const uint32_t id); 
  float gain(const uint16_t& strip)  const { return gainHandle->getStripGain( strip, gainRange ); }
  uint16_t applyGain(const uint16_t& strip,const uint16_t& adc );

  edm::EDGetTokenT<ClusterCollection> token;
  SiStripApvGain::Range gainRange;
  edm::ESHandle<SiStripGain> gainHandle;
  uint32_t gain_cache_id, detId;
  
};


SiStripClusterToDigiProducer::
SiStripClusterToDigiProducer(const edm::ParameterSet& conf) 
{
  
  token = consumes<ClusterCollection>(conf.getParameter<edm::InputTag>("ClusterProducer"));

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
  event.getByToken(token,input);

  if(input.isValid())
    process(*input, output_base);


  auto outputZS = std::make_unique<DigiCollection>(output_base);
  auto outputVR = std::make_unique<DigiCollection>();
  auto outputPR = std::make_unique<DigiCollection>();
  auto outputSM = std::make_unique<DigiCollection>();

  event.put(std::move(outputZS), "ZeroSuppressed");
  event.put(std::move(outputVR), "VirginRaw"     );
  event.put(std::move(outputPR), "ProcessedRaw"  );
  event.put(std::move(outputSM), "ScopeMode"     );
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
    
    if (!detDigis.empty()) 
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
