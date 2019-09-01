#ifndef HcalLutMetadataHandler_h
#define HcalLutMetadataHandler_h

/*
\class HcalLutMetadataHandler
\author Gena Kukartsev 21 Sep 2009
PopCon handler for the HCAL LUT metadata condition
*/

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
#include "CondFormats/HcalObjects/interface/HcalLutMetadata.h"
#include "CondFormats/DataRecord/interface/HcalLutMetadataRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

class HcalLutMetadataHandler : public popcon::PopConSourceHandler<HcalLutMetadata> {
public:
  void getNewObjects() override;
  std::string id() const override { return m_name; }
  ~HcalLutMetadataHandler() override;
  HcalLutMetadataHandler(edm::ParameterSet const&);

  void initObject(HcalLutMetadata*);

private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalLutMetadata* myDBObject;
  std::string m_name;
};
#endif
