#ifndef HcalCovarianceMatricesHandler_h
#define HcalCovarianceMatricesHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalCovarianceMatrices.h"
#include "CondFormats/DataRecord/interface/HcalCovarianceMatricesRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalCovarianceMatricesHandler : public popcon::PopConSourceHandler<HcalCovarianceMatrices>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~HcalCovarianceMatricesHandler();
  HcalCovarianceMatricesHandler(edm::ParameterSet const &);

  void initObject(HcalCovarianceMatrices*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalCovarianceMatrices* myDBObject;
  std::string m_name;

};
#endif
