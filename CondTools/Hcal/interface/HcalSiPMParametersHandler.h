#ifndef CondToolsHcalHcalSiPMParametersHandler_h
#define CondToolsHcalHcalSiPMParametersHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalSiPMParameters.h"
#include "CondFormats/DataRecord/interface/HcalSiPMParametersRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalSiPMParametersHandler : public popcon::PopConSourceHandler<HcalSiPMParameters> {

public:
  void getNewObjects() override;
  std::string id() const override { return m_name;}
  ~HcalSiPMParametersHandler() override;
  HcalSiPMParametersHandler(edm::ParameterSet const &);

  void initObject(HcalSiPMParameters*);
  
private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalSiPMParameters* myDBObject;
  std::string m_name;

};
#endif
