#ifndef HcalRecoParamsHandler_h
#define HcalRecoParamsHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"
#include "CondFormats/DataRecord/interface/HcalRecoParamsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalRecoParamsHandler : public popcon::PopConSourceHandler<HcalRecoParams>
{
 public:
  void getNewObjects() override;
  std::string id() const override { return m_name;}
  ~HcalRecoParamsHandler() override;
  HcalRecoParamsHandler(edm::ParameterSet const &);

  void initObject(HcalRecoParams*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalRecoParams* myDBObject;
  std::string m_name;

};
#endif
