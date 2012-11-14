#ifndef CastorSaturationCorrsHandler_h
#define CastorSaturationCorrsHandler_h



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
#include "CondFormats/CastorObjects/interface/CastorSaturationCorrs.h"
#include "CondFormats/DataRecord/interface/CastorSaturationCorrsRcd.h"
#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h"


class CastorSaturationCorrsHandler : public popcon::PopConSourceHandler<CastorSaturationCorrs>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~CastorSaturationCorrsHandler();
  CastorSaturationCorrsHandler(edm::ParameterSet const &);

  void initObject(CastorSaturationCorrs*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  CastorSaturationCorrs* myDBObject;
  std::string m_name;

};
#endif
