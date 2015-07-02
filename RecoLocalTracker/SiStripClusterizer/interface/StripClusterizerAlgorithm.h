#ifndef RecoLocalTracker_StripClusterizerAlgorithm_h
#define RecoLocalTracker_StripClusterizerAlgorithm_h

namespace edm{class EventSetup;}
class SiStripDigi;
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include <limits>


class FedChannelConnection;

class StripClusterizerAlgorithm {
  
 public:

  virtual ~StripClusterizerAlgorithm() {}
  virtual void initialize(const edm::EventSetup&);


  //Offline DetSet interface
  typedef edmNew::DetSetVector<SiStripCluster> output_t;
  void clusterize(const    edm::DetSetVector<SiStripDigi> &, output_t &);
  void clusterize(const edmNew::DetSetVector<SiStripDigi> &, output_t &);
  virtual void clusterizeDetUnit(const    edm::DetSet<SiStripDigi> &, output_t::FastFiller &) = 0;
  virtual void clusterizeDetUnit(const edmNew::DetSet<SiStripDigi> &, output_t::FastFiller &) = 0;

  //HLT stripByStrip interface
  virtual bool stripByStripBegin(uint32_t id) = 0;

  virtual void addFed(sistrip::FEDZSChannelUnpacker & unpacker, uint16_t ipair, std::vector<SiStripCluster>& out) {}
  virtual void stripByStripAdd(uint16_t strip, uint8_t adc, std::vector<SiStripCluster>& out) {}
  virtual void stripByStripEnd(std::vector<SiStripCluster>& out) {}

  virtual void addFed(sistrip::FEDZSChannelUnpacker & unpacker, uint16_t ipair, output_t::FastFiller & out) {}
  virtual void stripByStripAdd(uint16_t strip, uint8_t adc, output_t::FastFiller & out) {}
  virtual void stripByStripEnd(output_t::FastFiller & out) {}
  
  virtual void cleanState(){}

  struct InvalidChargeException : public cms::Exception { public: InvalidChargeException(const SiStripDigi&); };


  SiStripDetCabling const * cabling() const { return theCabling;} 
  std::vector<uint32_t> const & allDetIds() const { return detIds;}
  std::vector<const FedChannelConnection *> const & currentConnection() const { return connections[ind]; }

 protected:

  StripClusterizerAlgorithm() : qualityLabel(""), noise_cache_id(0), gain_cache_id(0), quality_cache_id(0) {}

  uint32_t currentId() {return detId;}
  bool setDetId(const uint32_t);
  float noise(const uint16_t& strip) const { return SiStripNoises::getNoise( strip, noiseRange ); }
  float gain(const uint16_t& strip)  const { return SiStripGain::getStripGain( strip, gainRange ); }
  bool bad(const uint16_t& strip)    const { return qualityHandle->IsStripBad( qualityRange, strip ); }
  bool isModuleBad(const uint32_t& id)  const { return qualityHandle->IsModuleBad( id ); }
  bool isModuleUsable(const uint32_t& id)  const { return qualityHandle->IsModuleUsable( id ); }
  bool allBadBetween(uint16_t L, const uint16_t& R) const { while( ++L < R  &&  bad(L) ); return L == R; }

  std::string qualityLabel;
  bool _setDetId;

 private:

  template<class T> void clusterize_(const T& input, output_t& output) {
    for(typename T::const_iterator it = input.begin(); it!=input.end(); it++) {
      output_t::FastFiller ff(output, it->detId());	
      clusterizeDetUnit(*it, ff);	
      if(ff.empty()) ff.abort();	
    }	
  }

  static constexpr unsigned short invalidI = std::numeric_limits<unsigned short>::max();
  struct Index { 
    unsigned short 
    gi=invalidI,
      ni=invalidI,
      qi=invalidI;
  };
  std::vector<uint32_t> detIds; // from cabling (connected and not bad)
  std::vector<std::vector<const FedChannelConnection *> > connections;
  std::vector<Index> indices;
  SiStripApvGain::Range gainRange;
  SiStripNoises::Range  noiseRange;
  SiStripQuality::Range qualityRange;
  edm::ESHandle<SiStripGain> gainHandle;
  edm::ESHandle<SiStripNoises> noiseHandle;
  edm::ESHandle<SiStripQuality> qualityHandle;
  SiStripDetCabling const * theCabling = nullptr;
  uint32_t noise_cache_id, gain_cache_id, quality_cache_id, detId=0;
  unsigned short ind=invalidI;

};
#endif
