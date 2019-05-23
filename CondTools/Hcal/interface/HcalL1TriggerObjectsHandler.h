#ifndef HcalL1TriggerObjectsHandler_h
#define HcalL1TriggerObjectsHandler_h

// Radek Ofierzynski, 9.11.2008

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
#include "CondFormats/HcalObjects/interface/HcalL1TriggerObjects.h"
#include "CondFormats/DataRecord/interface/HcalL1TriggerObjectsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

class HcalL1TriggerObjectsHandler : public popcon::PopConSourceHandler<HcalL1TriggerObjects> {
public:
  void getNewObjects() override;
  std::string id() const override { return m_name; }
  ~HcalL1TriggerObjectsHandler() override;
  HcalL1TriggerObjectsHandler(edm::ParameterSet const&);

  void initObject(HcalL1TriggerObjects*);

private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalL1TriggerObjects* myDBObject;
  std::string m_name;
};
#endif
