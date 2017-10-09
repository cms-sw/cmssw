#ifndef HcalMCParamsHandler_h
#define HcalMCParamsHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalMCParams.h"
#include "CondFormats/DataRecord/interface/HcalMCParamsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalMCParamsHandler : public popcon::PopConSourceHandler<HcalMCParams>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~HcalMCParamsHandler();
  HcalMCParamsHandler(edm::ParameterSet const &);

  void initObject(HcalMCParams*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalMCParams* myDBObject;
  std::string m_name;

};
#endif
