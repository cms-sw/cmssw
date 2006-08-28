#ifndef HCALTBTDCUNPACKER_H
#define HCALTBTDCUNPACKER_H 1
using namespace std;
#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

namespace hcaltb {
/** \class HcalTBTDCUnpacker
    
   $Date: 2006/08/04 00:28:00 $
   $Revision: 1.5 $
   \author J. Mans, P. Dudero - Minnesota
*/
class HcalTBTDCUnpacker {
public:
  HcalTBTDCUnpacker(bool);
  void unpack(const FEDRawData& raw,
	      HcalTBEventPosition& pos,
	      HcalTBTiming& timing) const;
  void setCalib(const vector<vector<string> >& calibLines_);
private:
  struct Hit {
    int channel;
    double time;
  };
  
  void unpackHits(const FEDRawData& raw, std::vector<Hit>& hits, HcalTBTiming& timing) const;
  void reconstructWC(const std::vector<Hit>& hits,
		     HcalTBEventPosition& pos) const;
  void reconstructTiming(const std::vector<Hit>& hits,
			 HcalTBTiming& timing) const;
//  void setupWC();  // reads it from configuration file

  static const int PLANECOUNT = 16;
  static const int WC_CHANNELIDS[PLANECOUNT*3];
  struct WireChamberRecoData {
    double b0, b1, mean, sigma;
  } wc_[PLANECOUNT];

  bool includeUnmatchedHits_;
  double tdc_ped[161];
  double tdc_convers[161];
};

}

#endif
