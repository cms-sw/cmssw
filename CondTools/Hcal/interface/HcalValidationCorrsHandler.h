#ifndef HcalRespCorrsHandler_h
#define HcalRespCorrsHandler_h

// Gena Kukartsev, 29.07.2009


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
#include "CondFormats/HcalObjects/interface/HcalValidationCorrs.h"
#include "CondFormats/DataRecord/interface/HcalValidationCorrsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalValidationCorrsHandler : public popcon::PopConSourceHandler<HcalValidationCorrs>
{
 public:
  void getNewObjects() override;
  std::string id() const override { return m_name;}
  ~HcalValidationCorrsHandler() override;
  HcalValidationCorrsHandler(edm::ParameterSet const &);

  void initObject(HcalValidationCorrs*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalValidationCorrs* myDBObject;
  std::string m_name;

};
#endif
