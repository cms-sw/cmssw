#ifndef CondToolsHcalHcalHcalTPParametersHandler_h
#define CondToolsHcalHcalHcalTPParametersHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalTPParameters.h"
#include "CondFormats/DataRecord/interface/HcalTPParametersRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

class HcalTPParametersHandler : public popcon::PopConSourceHandler<HcalTPParameters> {
public:
  void getNewObjects() override;
  std::string id() const override { return m_name; }
  ~HcalTPParametersHandler() override;
  HcalTPParametersHandler(edm::ParameterSet const&);

  void initObject(HcalTPParameters*);

private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalTPParameters* myDBObject;
  std::string m_name;
};
#endif
