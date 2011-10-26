#ifndef HcalTimingParamsHandler_h
#define HcalTimingParamsHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalTimingParams.h"
#include "CondFormats/DataRecord/interface/HcalTimingParamsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalTimingParamsHandler : public popcon::PopConSourceHandler<HcalTimingParams>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~HcalTimingParamsHandler();
  HcalTimingParamsHandler(edm::ParameterSet const &);

  void initObject(HcalTimingParams*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalTimingParams* myDBObject;
  std::string m_name;

};
#endif
