/* -*- C++ -*- */
#ifndef HcalTBSlowDataUnpacker_h_included
#define HcalTBSlowDataUnpacker_h_included 1

#include "TBDataFormats/HcalTBObjects/interface/HcalTBRunData.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBEventPosition.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include <map>
#include <string>

namespace hcaltb {
  /** \brief Unpacks "SlowData" packages used for SiPM calibration and other purposes in HCAL Local DAQ.

      To use this code in an analysis module, you need only:

      analyze(const edm::Event& iEvent, const edm::EventSetup&) {
      
      edm::Handle<edm::FEDRawDataCollection> rawraw;  
      iEvent.getByType(rawraw);        

      hcaltb::HcalTBSlowDataUnpacker sdp;
      std::map<std::string,std::string> strings;
      std::map<std::string,double> numerics;
      // if regular slow data
      sdp.unpackMaps(rawraw->FEDData(hcaltb::HcalTBSlowDataUnpacker::STANDARD_FED_ID),strings,numerics);
      // if SiPM Cal slow data (different 'FED')
      sdp.unpackMaps(rawraw->FEDData(hcaltb::HcalTBSlowDataUnpacker::SIPM_CAL_FED_ID),strings,numerics);
      
  */
  class HcalTBSlowDataUnpacker {
  public:
    HcalTBSlowDataUnpacker(void) { }

    void unpack(const FEDRawData&    raw,
		HcalTBRunData&            htbrd,
		HcalTBEventPosition&      htbep) const;

    void unpackMaps(const FEDRawData&    raw, std::map<std::string,std::string>& strings, std::map<std::string,double>& numerics) const;

    static const int STANDARD_FED_ID=3;
    static const int SIPM_CAL_FED_ID=11;
  };
}

#endif // HcalTBSlowDataUnpacker_h_included
