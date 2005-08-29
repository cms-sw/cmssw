#ifndef HCALTBTDCUNPACKER_H
#define HCALTBTDCUNPACKER_H 1

#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

/** \class HcalTBTDCUnpacker
    
   $Date: $
   $Revision: $
   \author J. Mans, P. Dudero - Minnesota
*/
class HcalTBTDCUnpacker {
public:
  HcalTBTDCUnpacker();
  void unpack(const raw::FEDRawData& raw,
	      hcaltb::HcalTBEventPosition& pos,
	      hcaltb::HcalTBTiming& timing) const;
private:
  struct Hit {
    int channel;
    double time;
  };
  
  void unpackHits(const raw::FEDRawData& raw, std::vector<Hit>& hits) const;
  void reconstructWC(const std::vector<Hit>& hits,
		     hcaltb::HcalTBEventPosition& pos) const;
  void reconstructTiming(const std::vector<Hit>& hits,
			 hcaltb::HcalTBTiming& timing) const;
  void setupWC();

  static const int PLANECOUNT = 10;
  static const int WC_CHANNELIDS[PLANECOUNT*3];
  struct WireChamberRecoData {
    double b0, b1, mean, sigma;
  } wc_[PLANECOUNT];

  

};
#endif
