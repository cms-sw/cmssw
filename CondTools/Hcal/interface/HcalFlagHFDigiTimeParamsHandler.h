#ifndef HcalFlagHFDigiTimeParamsHandler_h
#define HcalFlagHFDigiTimeParamsHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalFlagHFDigiTimeParams.h"
#include "CondFormats/DataRecord/interface/HcalFlagHFDigiTimeParamsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalFlagHFDigiTimeParamsHandler : public popcon::PopConSourceHandler<HcalFlagHFDigiTimeParams>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~HcalFlagHFDigiTimeParamsHandler();
  HcalFlagHFDigiTimeParamsHandler(edm::ParameterSet const &);

  void initObject(HcalFlagHFDigiTimeParams*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalFlagHFDigiTimeParams* myDBObject;
  std::string m_name;

};
#endif
