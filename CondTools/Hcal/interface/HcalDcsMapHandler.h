#ifndef HcalDcsMapHandler_h
#define HcalDcsMapHandler_h

// Gena Kukartsev, February 5, 2010


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
#include "CondFormats/HcalObjects/interface/HcalDcsMap.h"
#include "CondFormats/DataRecord/interface/HcalDcsMapRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalDcsMapHandler : public popcon::PopConSourceHandler<HcalDcsMap>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~HcalDcsMapHandler();
  HcalDcsMapHandler(edm::ParameterSet const &);

  void initObject(HcalDcsMap*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalDcsMap* myDBObject;
  std::string m_name;

};
#endif
