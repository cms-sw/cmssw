#ifndef HcalElectronicsMapHandler_h
#define HcalElectronicsMapHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalElectronicsMapHandler : public popcon::PopConSourceHandler<HcalElectronicsMap>
{
 public:
  void getNewObjects() override;
  std::string id() const override { return m_name;}
  ~HcalElectronicsMapHandler() override;
  HcalElectronicsMapHandler(edm::ParameterSet const &);

  void initObject(HcalElectronicsMap*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalElectronicsMap* myDBObject;
  std::string m_name;

};
#endif
