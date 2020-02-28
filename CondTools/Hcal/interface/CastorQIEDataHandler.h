#ifndef CastorQIEDataHandler_h
#define CastorQIEDataHandler_h

// Radek Ofierzynski, 27.02.2008
// Adapted for CASTOR by L. Mundim (26/03/2009)

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
#include "CondFormats/CastorObjects/interface/CastorQIEData.h"
#include "CondFormats/DataRecord/interface/CastorQIEDataRcd.h"
#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h"

class CastorQIEDataHandler : public popcon::PopConSourceHandler<CastorQIEData> {
public:
  void getNewObjects() override;
  std::string id() const override { return m_name; }
  ~CastorQIEDataHandler() override;
  CastorQIEDataHandler(edm::ParameterSet const&);

  void initObject(CastorQIEData*);

private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  CastorQIEData* myDBObject;
  std::string m_name;
};
#endif
