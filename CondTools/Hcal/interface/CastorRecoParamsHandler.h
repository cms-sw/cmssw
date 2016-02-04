#ifndef CastorRecoParamsHandler_h
#define CastorRecoParamsHandler_h

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
#include "CondFormats/CastorObjects/interface/CastorRecoParams.h"
#include "CondFormats/DataRecord/interface/CastorRecoParamsRcd.h"
#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h"


class CastorRecoParamsHandler : public popcon::PopConSourceHandler<CastorRecoParams>
{
 public:
  void getNewObjects();
  std::string id() const { return m_name;}
  ~CastorRecoParamsHandler();
  CastorRecoParamsHandler(edm::ParameterSet const &);

  void initObject(CastorRecoParams*);

 private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  CastorRecoParams* myDBObject;
  std::string m_name;

};
#endif
