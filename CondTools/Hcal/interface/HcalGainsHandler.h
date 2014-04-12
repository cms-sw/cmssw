#ifndef HcalGainsHandler_h
#define HcalGainsHandler_h

// Radek Ofierzynski, 27.02.2008


#include <string>
#include <iostream>
#include <typeinfo>
#include <fstream>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
 
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
// user include files
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalGainsHandler : public popcon::PopConSourceHandler<HcalGains>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~HcalGainsHandler();
  HcalGainsHandler(edm::ParameterSet const &);

  void initObject(HcalGains*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalGains* myDBObject;
  std::string m_name;

};
#endif
