#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripZeroSuppressor.h"

using namespace std;

void SiStripZeroSuppressor::suppress(const edm::DetSet<SiStripRawDigi>& in, edm::DetSet<SiStripDigi>& out)
{

  edm::DetSet<SiStripRawDigi>::const_iterator in_iter=in.data.begin();
  for (uint i=0; i<in.data.size(); i++){
    out.data.push_back(SiStripDigi(i, in_iter->adc()));
    in_iter++;
  }

}

void SiStripZeroSuppressor::suppress(const std::vector<int16_t>& in, edm::DetSet<SiStripDigi>& out)
{

  edm::LogError("SiStrip") << "Zero suppression: det size = " << in.size() << std::endl;

  for (uint i=0; i<in.size(); i++){
    edm::LogError("SiStrip") << "strip = " << i << "  adc = " << in[i] << std::endl;
    out.data.push_back(SiStripDigi(i, in[i]));
  }
}
