#ifndef HcalGainWidthsHandler_h
#define HcalGainWidthsHandler_h

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
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"


class HcalGainWidthsHandler : public popcon::PopConSourceHandler<HcalGainWidths>
{
 public:
  void getNewObjects() override;
  std::string id() const override { return m_name;}
  ~HcalGainWidthsHandler() override;
  HcalGainWidthsHandler(edm::ParameterSet const &);

  void initObject(HcalGainWidths*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalGainWidths* myDBObject;
  std::string m_name;

};
#endif
