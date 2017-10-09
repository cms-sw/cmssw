#ifndef HcalQIEDataHandler_h
#define HcalQIEDataHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalQIEDataHandler : public popcon::PopConSourceHandler<HcalQIEData>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~HcalQIEDataHandler();
  HcalQIEDataHandler(edm::ParameterSet const &);

  void initObject(HcalQIEData*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalQIEData* myDBObject;
  std::string m_name;

};
#endif
