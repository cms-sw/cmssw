#ifndef HcalChannelQualityHandler_h
#define HcalChannelQualityHandler_h

// Radek Ofierzynski, 27.02.2008


#include <string>
#include <iostream>
#include <typeinfo>
#include <fstream>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondCore/PopCon/interface/LogReader.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
// user include files
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalChannelQualityHandler : public popcon::PopConSourceHandler<HcalChannelQuality>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~HcalChannelQualityHandler();
  HcalChannelQualityHandler(edm::ParameterSet const &);

 private:
  unsigned int sinceTime;
  std::string fFile;

  std::string m_name;

};
#endif
