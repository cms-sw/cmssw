#ifndef HcalPedestalsHandler_h
#define HcalPedestalsHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

class HcalPedestalsHandler : public popcon::PopConSourceHandler<HcalPedestals> {
public:
  void getNewObjects() override;
  std::string id() const override { return m_name; }
  ~HcalPedestalsHandler() override;
  HcalPedestalsHandler(edm::ParameterSet const&);

  void initObject(HcalPedestals*);

private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalPedestals* myDBObject;
  std::string m_name;
};
#endif
