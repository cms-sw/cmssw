#ifndef HcalPedestalWidthsHandler_h
#define HcalPedestalWidthsHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalPedestalWidthsHandler : public popcon::PopConSourceHandler<HcalPedestalWidths>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~HcalPedestalWidthsHandler();
  HcalPedestalWidthsHandler(edm::ParameterSet const &);

  void initObject(HcalPedestalWidths*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalPedestalWidths* myDBObject;
  std::string m_name;

};
#endif
