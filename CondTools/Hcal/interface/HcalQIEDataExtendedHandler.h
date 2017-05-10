#ifndef HcalQIEDataExtendedHandler_h
#define HcalQIEDataExtendedHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalQIEDataExtended.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataExtendedRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalQIEDataExtendedHandler : public popcon::PopConSourceHandler<HcalQIEDataExtended>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~HcalQIEDataExtendedHandler();
  HcalQIEDataExtendedHandler(edm::ParameterSet const &);

  void initObject(HcalQIEDataExtended*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalQIEDataExtended* myDBObject;
  std::string m_name;

};
#endif
