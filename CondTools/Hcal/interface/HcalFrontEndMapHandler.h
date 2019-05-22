#ifndef HcalFrontEndMapHandler_h
#define HcalFrontEndMapHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalFrontEndMap.h"
#include "CondFormats/DataRecord/interface/HcalFrontEndMapRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

class HcalFrontEndMapHandler : public popcon::PopConSourceHandler<HcalFrontEndMap> {
public:
  void getNewObjects() override;
  std::string id() const override { return m_name; }
  ~HcalFrontEndMapHandler() override;
  HcalFrontEndMapHandler(edm::ParameterSet const&);

  void initObject(HcalFrontEndMap*);

private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalFrontEndMap* myDBObject;
  std::string m_name;
};
#endif
