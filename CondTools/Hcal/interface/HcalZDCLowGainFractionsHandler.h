#ifndef HcalZDCLowGainFractionsHandler_h
#define HcalZDCLowGainFractionsHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalZDCLowGainFractions.h"
#include "CondFormats/DataRecord/interface/HcalZDCLowGainFractionsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalZDCLowGainFractionsHandler : public popcon::PopConSourceHandler<HcalZDCLowGainFractions>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~HcalZDCLowGainFractionsHandler();
  HcalZDCLowGainFractionsHandler(edm::ParameterSet const &);

  void initObject(HcalZDCLowGainFractions*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalZDCLowGainFractions* myDBObject;
  std::string m_name;

};
#endif
