#ifndef RPCDBHandler_h
#define RPCDBHandler_h

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>
#include <iostream>
#include <sstream>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/RPCObjects/interface/RPCDQMObject.h"
#include "CondFormats/DataRecord/interface/RPCDQMObjectRcd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "DQM/RPCMonitorDigi/interface/utils.h"


class RPCDBHandler : public popcon::PopConSourceHandler<RPCDQMObject>
{
      
public:
  void getNewObjects();
  std::string id() const { return m_name; }
  ~RPCDBHandler(); 
  RPCDBHandler(const edm::ParameterSet& pset);
     
  void initObject(RPCDQMObject*);
 
private:
  std::string m_name;
  unsigned int sinceTime;
  RPCDQMObject * rpcDQMObject;
};

#endif
