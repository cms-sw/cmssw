#ifndef HcalDcsValuesHandler_h
#define HcalDcsValuesHandler_h

// Jake Anderson, 20.10.2009

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
#include "CondFormats/HcalObjects/interface/HcalDcsValues.h"
#include "CondFormats/DataRecord/interface/HcalDcsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

class HcalDcsValuesHandler : public popcon::PopConSourceHandler<HcalDcsValues> {
public:
  void getNewObjects() override;
  std::string id() const override { return m_name; }
  ~HcalDcsValuesHandler() override;
  HcalDcsValuesHandler(edm::ParameterSet const&);

  void initObject(HcalDcsValues*);

private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalDcsValues* myDBObject;
  std::string m_name;
};
#endif
