#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPPEDESTALSSUBTRACTOR_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPPEDESTALSSUBTRACTOR_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include <vector>

class SiStripPedestalsSubtractor {

  friend class SiStripRawProcessingFactory;

 public:

  void subtract(const edm::DetSet<SiStripRawDigi>& input, std::vector<int16_t>& output);
  void subtract(uint32_t detId, uint16_t firstAPV, std::vector<int16_t>& adcs);
  void init(const edm::EventSetup&);

 private:

  SiStripPedestalsSubtractor(bool mode) : peds_cache_id(0), fedmode_(mode) {};
  edm::ESHandle<SiStripPedestals> pedestalsHandle;
  std::vector<int> pedestals;
  uint32_t peds_cache_id;
  bool fedmode_;

  template <class input_t> void subtract_(uint32_t detId, uint16_t firstAPV, const input_t& input, std::vector<int16_t>& output);
  int16_t eval(int16_t in) { return in; }
  uint16_t eval(SiStripRawDigi in) { return in.adc(); }

};
#endif
