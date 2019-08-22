#ifndef CastorChannelQualityHandler_h
#define CastorChannelQualityHandler_h

// Radek Ofierzynski, 27.02.2008
// Adapter for CASTOR by L. Mundim (26/03/2009)

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
#include "CondFormats/CastorObjects/interface/CastorChannelQuality.h"
#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h"
#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h"

class CastorChannelQualityHandler : public popcon::PopConSourceHandler<CastorChannelQuality> {
public:
  void getNewObjects() override;
  std::string id() const override { return m_name; }
  ~CastorChannelQualityHandler() override;
  CastorChannelQualityHandler(edm::ParameterSet const&);

  void initObject(CastorChannelQuality*);

private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  CastorChannelQuality* myDBObject;
  std::string m_name;
};
#endif
