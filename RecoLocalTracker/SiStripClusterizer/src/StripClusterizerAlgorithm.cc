#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void StripClusterizerAlgorithm::clusterize(const edm::DetSetVector<SiStripDigi>& input, output_t& output) const {
  clusterize_(input, output);
}
void StripClusterizerAlgorithm::clusterize(const edmNew::DetSetVector<SiStripDigi>& input, output_t& output) const {
  clusterize_(input, output);
}

StripClusterizerAlgorithm::InvalidChargeException::InvalidChargeException(const SiStripDigi& digi)
    : cms::Exception("Invalid Charge") {
  std::stringstream s;
  s << "Digi charge of " << digi.adc() << " ADC "
    << "is out of range on strip " << digi.strip() << ".  ";
  this->append(s.str());
}
