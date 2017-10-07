#ifndef HcalLUTCorrsHandler_h
#define HcalLUTCorrsHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalLUTCorrs.h"
#include "CondFormats/DataRecord/interface/HcalLUTCorrsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalLUTCorrsHandler : public popcon::PopConSourceHandler<HcalLUTCorrs>
{
 public:
  void getNewObjects() override;
  std::string id() const override { return m_name;}
  ~HcalLUTCorrsHandler() override;
  HcalLUTCorrsHandler(edm::ParameterSet const &);

  void initObject(HcalLUTCorrs*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalLUTCorrs* myDBObject;
  std::string m_name;

};
#endif
