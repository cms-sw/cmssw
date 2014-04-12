#ifndef HcalCholeskyMatricesHandler_h
#define HcalCholeskyMatricesHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalCholeskyMatrices.h"
#include "CondFormats/DataRecord/interface/HcalCholeskyMatricesRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalCholeskyMatricesHandler : public popcon::PopConSourceHandler<HcalCholeskyMatrices>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~HcalCholeskyMatricesHandler();
  HcalCholeskyMatricesHandler(edm::ParameterSet const &);

  void initObject(HcalCholeskyMatrices*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalCholeskyMatrices* myDBObject;
  std::string m_name;

};
#endif
