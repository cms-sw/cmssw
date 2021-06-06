#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPPEDESTALSSUBTRACTOR_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPPEDESTALSSUBTRACTOR_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include <vector>

class SiStripPedestalsSubtractor {
  friend class SiStripRawProcessingFactory;

public:
  void subtract(const edm::DetSet<SiStripRawDigi>& input, std::vector<int16_t>& output);
  void subtract(uint32_t detId, uint16_t firstAPV, std::vector<int16_t>& adcs);
  void init(const edm::EventSetup&);

private:
  SiStripPedestalsSubtractor(bool mode, edm::ConsumesCollector iC)
      : pedestalsToken_(iC.esConsumes<SiStripPedestals, SiStripPedestalsRcd>()), fedmode_(mode) {}
  edm::ESGetToken<SiStripPedestals, SiStripPedestalsRcd> pedestalsToken_;
  edm::ESWatcher<SiStripPedestalsRcd> pedestalsWatcher_;
  const SiStripPedestals* pedestalsHandle;
  std::vector<int> pedestals;
  bool fedmode_;

  template <class input_t>
  void subtract_(uint32_t detId, uint16_t firstAPV, const input_t& input, std::vector<int16_t>& output);
  int16_t eval(int16_t in) { return in; }
  uint16_t eval(SiStripRawDigi in) { return in.adc(); }
};
#endif
