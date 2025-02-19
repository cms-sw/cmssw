#include "RecoLocalTracker/SiStripClusterizer/test/ClusterizerUnitTesterESProducer.h"

ClusterizerUnitTesterESProducer::
ClusterizerUnitTesterESProducer(const edm::ParameterSet& conf) :
  apvGain(new SiStripApvGain()),
  noises(new SiStripNoises())
{
  edm::FileInPath testInfo(("RecoLocalTracker/SiStripClusterizer/test/ClusterizerUnitTestDetInfo.dat"));
  quality = boost::shared_ptr<SiStripQuality>(new SiStripQuality(testInfo));

  extractNoiseGainQuality(conf);

  gain = boost::shared_ptr<SiStripGain>(new SiStripGain(*apvGain,1));

  quality->cleanUp();
  quality->fillBadComponents();

  setWhatProduced(this,&ClusterizerUnitTesterESProducer::produceGainRcd);
  setWhatProduced(this,&ClusterizerUnitTesterESProducer::produceNoisesRcd);
  setWhatProduced(this,&ClusterizerUnitTesterESProducer::produceQualityRcd);
}

void 
ClusterizerUnitTesterESProducer::
extractNoiseGainQuality(const edm::ParameterSet& conf) {
  uint32_t detId = 0;
  VPSet groups = conf.getParameter<VPSet>("ClusterizerTestGroups");
  for(iter_t group = groups.begin(); group < groups.end(); group++) {
    VPSet tests = group->getParameter<VPSet>("Tests");
    for(iter_t test = tests.begin();  test < tests.end();  test++)
      extractNoiseGainQualityForDetId( detId++, test->getParameter<VPSet>("Digis"));
  }
}

void
ClusterizerUnitTesterESProducer::
extractNoiseGainQualityForDetId(uint32_t detId, const VPSet& digiset) {
  std::vector<std::pair<uint16_t,float> > detNoises;
  std::vector<std::pair<uint16_t,float> > detGains;
  std::vector<unsigned> detBadStrips;
  for(iter_t digi = digiset.begin(); digi<digiset.end(); digi++) {
    uint16_t strip = digi->getParameter<unsigned>("Strip");
    if(digi->getParameter<unsigned>("ADC") != 0) {
      detNoises.push_back(std::make_pair(strip, digi->getParameter<double>("Noise") ));
      detGains.push_back(std::make_pair(strip, digi->getParameter<double>("Gain") ));
    }
    if(!digi->getParameter<bool>("Quality") )
      detBadStrips.push_back(quality->encode(strip,1));
  }
  setNoises(detId, detNoises);
  setGains(detId, detGains);
  if(detBadStrips.size())
    quality->add(detId, std::make_pair(detBadStrips.begin(), detBadStrips.end()));  
}


void 
ClusterizerUnitTesterESProducer::
setNoises(uint32_t detId, std::vector<std::pair<uint16_t, float> >& digiNoises ) {
  std::sort(digiNoises.begin(),digiNoises.end());
  std::vector<float> detnoise;
  for(std::vector<std::pair<uint16_t,float> >::const_iterator digi=digiNoises.begin(); digi<digiNoises.end(); ++digi){
    detnoise.resize(digi->first,1); //pad with default noise 1
    detnoise.push_back(digi->second);
  }
  if(detnoise.size() > 768) throw cms::Exception("Faulty noise construction") << "No strip numbers greater than 767 please" << std::endl;
  detnoise.resize(768,1.0);
  
  SiStripNoises::InputVector theSiStripVector;
  for(uint16_t strip=0; strip<detnoise.size(); strip++) {
    noises->setData( detnoise.at(strip), theSiStripVector);
  }
  noises->put(detId,theSiStripVector);
}

void 
ClusterizerUnitTesterESProducer::
setGains(uint32_t detId, std::vector<std::pair<uint16_t, float> >& digiGains) {
  std::sort(digiGains.begin(),digiGains.end());
  std::vector<float> detApvGains;
  for(std::vector<std::pair<uint16_t,float> >::const_iterator digi=digiGains.begin(); digi<digiGains.end(); ++digi){
    if( detApvGains.size() <= digi->first / 128 ) {
      detApvGains.push_back( digi->second );
    }
    else if(detApvGains.at( digi->first / 128 ) != digi->second) {
      throw cms::Exception("Faulty gain construction.") << "  Only one gain setting per APV please.\n";
    }
  }
  detApvGains.resize(6,1.);
  
  SiStripApvGain::Range range(detApvGains.begin(),detApvGains.end());
  if ( ! apvGain->put(detId,range) ) 
    throw cms::Exception("Trying to set gain twice for same detId: ") << detId;
  return;
}
