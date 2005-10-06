#ifndef HCALTBTDCUNPACKER_H
#define HCALTBTDCUNPACKER_H 1

#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

namespace hcaltb {
/** \class HcalTBTDCUnpacker
    
   $Date: 2005/08/29 17:31:59 $
   $Revision: 1.1 $
   \author J. Mans, P. Dudero - Minnesota
*/
class HcalTBTDCUnpacker {
public:
  HcalTBTDCUnpacker(bool);
  void unpack(const FEDRawData& raw,
	      HcalTBEventPosition& pos,
	      HcalTBTiming& timing) const;
private:
  struct Hit {
    int channel;
    double time;
  };
  
  void unpackHits(const FEDRawData& raw, std::vector<Hit>& hits) const;
  void reconstructWC(const std::vector<Hit>& hits,
		     HcalTBEventPosition& pos) const;
  void reconstructTiming(const std::vector<Hit>& hits,
			 HcalTBTiming& timing) const;
  void setupWC();

  static const int PLANECOUNT = 10;
  static const int WC_CHANNELIDS[PLANECOUNT*3];
  struct WireChamberRecoData {
    double b0, b1, mean, sigma;
  } wc_[PLANECOUNT];

  bool includeUnmatchedHits_;

};

}

#endif
