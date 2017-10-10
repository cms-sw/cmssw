#ifndef CondToolsHcalHcalSiPMCharacteristicsHandler_h
#define CondToolsHcalHcalSiPMCharacteristicsHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristics.h"
#include "CondFormats/DataRecord/interface/HcalSiPMCharacteristicsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalSiPMCharacteristicsHandler : public popcon::PopConSourceHandler<HcalSiPMCharacteristics> {

public:
  void getNewObjects() override;
  std::string id() const override { return m_name;}
  ~HcalSiPMCharacteristicsHandler() override;
  HcalSiPMCharacteristicsHandler(edm::ParameterSet const &);

  void initObject(HcalSiPMCharacteristics*);

private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalSiPMCharacteristics* myDBObject;
  std::string m_name;

};
#endif
