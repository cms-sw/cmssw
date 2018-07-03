#ifndef HcalLongRecoParamsHandler_h
#define HcalLongRecoParamsHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalLongRecoParams.h"
#include "CondFormats/DataRecord/interface/HcalLongRecoParamsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalLongRecoParamsHandler : public popcon::PopConSourceHandler<HcalLongRecoParams>
{
 public:
  void getNewObjects() override;
  std::string id() const override { return m_name;}
  ~HcalLongRecoParamsHandler() override;
  HcalLongRecoParamsHandler(edm::ParameterSet const &);

  void initObject(HcalLongRecoParams*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalLongRecoParams* myDBObject;
  std::string m_name;

};
#endif
