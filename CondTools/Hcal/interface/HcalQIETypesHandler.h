#ifndef HcalQIETypesHandler_h
#define HcalQIETypesHandler_h

// Walter Alda, 11.10.2015


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
#include "CondFormats/HcalObjects/interface/HcalQIETypes.h"
#include "CondFormats/DataRecord/interface/HcalQIETypesRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalQIETypesHandler : public popcon::PopConSourceHandler<HcalQIETypes>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~HcalQIETypesHandler();
  HcalQIETypesHandler(edm::ParameterSet const &);

  void initObject(HcalQIETypes*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalQIETypes* myDBObject;
  std::string m_name;

};
#endif
