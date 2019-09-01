#ifndef CondToolsHcalHcalTPChannelParametersHandler_h
#define CondToolsHcalHcalTPChannelParametersHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalTPChannelParameters.h"
#include "CondFormats/DataRecord/interface/HcalTPChannelParametersRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

class HcalTPChannelParametersHandler : public popcon::PopConSourceHandler<HcalTPChannelParameters> {
public:
  void getNewObjects() override;
  std::string id() const override { return m_name; }
  ~HcalTPChannelParametersHandler() override;
  HcalTPChannelParametersHandler(edm::ParameterSet const&);

  void initObject(HcalTPChannelParameters*);

private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalTPChannelParameters* myDBObject;
  std::string m_name;
};
#endif
