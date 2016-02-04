#ifndef HcalZSThresholdsHandler_h
#define HcalZSThresholdsHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalZSThresholds.h"
#include "CondFormats/DataRecord/interface/HcalZSThresholdsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalZSThresholdsHandler : public popcon::PopConSourceHandler<HcalZSThresholds>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~HcalZSThresholdsHandler();
  HcalZSThresholdsHandler(edm::ParameterSet const &);

  void initObject(HcalZSThresholds*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalZSThresholds* myDBObject;
  std::string m_name;

};
#endif
