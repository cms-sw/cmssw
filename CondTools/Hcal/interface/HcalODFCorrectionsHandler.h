#ifndef HcalODFCorrectionsHandler_h
#define HcalODFCorrectionsHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalODFCorrections.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalODFCorrectionsHandler : public popcon::PopConSourceHandler<HcalODFCorrections>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~HcalODFCorrectionsHandler();
  HcalODFCorrectionsHandler(edm::ParameterSet const &);

  void initObject(HcalODFCorrections*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalODFCorrections* myDBObject;
  std::string m_name;

};
#endif
