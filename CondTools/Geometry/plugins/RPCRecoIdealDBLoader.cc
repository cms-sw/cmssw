#include "RPCRecoIdealDBLoader.h"

#include <Geometry/RPCGeometryBuilder/src/RPCGeometryParsFromDD.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "Geometry/Records/interface/RPCRecoGeometryRcd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
using namespace std;

RPCRecoIdealDBLoader::RPCRecoIdealDBLoader(const edm::ParameterSet& iConfig) : label_()
{
  std::cout<<"RPCRecoIdealDBLoader::RPCRecoIdealDBLoader"<<std::endl;
}

RPCRecoIdealDBLoader::~RPCRecoIdealDBLoader()
{
  std::cout<<"RPCRecoIdealDBLoader::~RPCRecoIdealDBLoader"<<std::endl;
}

void
RPCRecoIdealDBLoader::beginRun( const edm::Run&, edm::EventSetup const& es) 
{
  RecoIdealGeometry* rig = new RecoIdealGeometry;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    edm::LogError("RPCRecoIdealDBLoader")<<"PoolDBOutputService unavailable";
    return;
  }

  edm::ESTransientHandle<DDCompactView> pDD;
  edm::ESHandle<MuonDDDConstants> pMNDC;
  es.get<IdealGeometryRecord>().get(label_, pDD );
  es.get<MuonNumberingRecord>().get( pMNDC );

  const DDCompactView& cpv = *pDD;
  RPCGeometryParsFromDD rpcpd;

  rpcpd.build( &cpv, *pMNDC, *rig );

  if ( mydbservice->isNewTagRequest("RPCRecoGeometryRcd") ) {
    mydbservice->createNewIOV<RecoIdealGeometry>(rig
                                                 , mydbservice->beginOfTime()
                                                 , mydbservice->endOfTime()
                                                 , "RPCRecoGeometryRcd");
  } else {
    edm::LogError("RPCRecoIdealDBLoader")<<"RPCRecoGeometryRcd Tag is already present.";
  }
}
