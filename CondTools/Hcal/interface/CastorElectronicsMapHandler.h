#ifndef CastorElectronicsMapHandler_h
#define CastorElectronicsMapHandler_h

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
#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"
#include "CondFormats/DataRecord/interface/CastorElectronicsMapRcd.h"
#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h"


class CastorElectronicsMapHandler : public popcon::PopConSourceHandler<CastorElectronicsMap>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~CastorElectronicsMapHandler();
  CastorElectronicsMapHandler(edm::ParameterSet const &);

  void initObject(CastorElectronicsMap*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  CastorElectronicsMap* myDBObject;
  std::string m_name;

};
#endif
