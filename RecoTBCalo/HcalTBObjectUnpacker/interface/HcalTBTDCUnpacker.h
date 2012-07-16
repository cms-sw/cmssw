#ifndef HCALTBTDCUNPACKER_H
#define HCALTBTDCUNPACKER_H 1
#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTiming.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

namespace hcaltb {
/** \class HcalTBTDCUnpacker
    
   $Date: 2010/10/19 18:29:26 $
   $Revision: 1.7 $
   \author J. Mans, P. Dudero - Minnesota
*/
class HcalTBTDCUnpacker {
public:
  HcalTBTDCUnpacker(bool);
  void unpack(const FEDRawData& raw,
	      HcalTBEventPosition& pos,
	      HcalTBTiming& timing) const;
  void setCalib(const std::vector<std::vector<std::string> >& calibLines_);
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
  FILE   *dumpObs_;
};

}

#endif
