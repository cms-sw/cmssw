#ifndef HcalTimeCorrsHandler_h
#define HcalTimeCorrsHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalTimeCorrs.h"
#include "CondFormats/DataRecord/interface/HcalTimeCorrsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalTimeCorrsHandler : public popcon::PopConSourceHandler<HcalTimeCorrs>
{
 public:
  void getNewObjects() override;
  std::string id() const override { return m_name;}
  ~HcalTimeCorrsHandler() override;
  HcalTimeCorrsHandler(edm::ParameterSet const &);

  void initObject(HcalTimeCorrs*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalTimeCorrs* myDBObject;
  std::string m_name;

};
#endif
