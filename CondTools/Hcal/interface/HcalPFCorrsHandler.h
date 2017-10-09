#ifndef HcalPFCorrsHandler_h
#define HcalPFCorrsHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalPFCorrs.h"
#include "CondFormats/DataRecord/interface/HcalPFCorrsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalPFCorrsHandler : public popcon::PopConSourceHandler<HcalPFCorrs>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~HcalPFCorrsHandler();
  HcalPFCorrsHandler(edm::ParameterSet const &);

  void initObject(HcalPFCorrs*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalPFCorrs* myDBObject;
  std::string m_name;

};
#endif
