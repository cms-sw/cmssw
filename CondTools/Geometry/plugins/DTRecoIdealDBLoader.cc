#include "DTRecoIdealDBLoader.h"

#include <Geometry/DTGeometryBuilder/src/DTGeometryParsFromDD.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "Geometry/Records/interface/DTRecoGeometryRcd.h"
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

DTRecoIdealDBLoader::DTRecoIdealDBLoader(const edm::ParameterSet& iConfig) : label_()
{
  std::cout<<"DTRecoIdealDBLoader::DTRecoIdealDBLoader"<<std::endl;
}

DTRecoIdealDBLoader::~DTRecoIdealDBLoader()
{
  std::cout<<"DTRecoIdealDBLoader::~DTRecoIdealDBLoader"<<std::endl;
}

void
DTRecoIdealDBLoader::beginRun( const edm::Run&, edm::EventSetup const& es) 
{
  RecoIdealGeometry* rig = new RecoIdealGeometry;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    edm::LogError("DTRecoIdealDBLoader")<<"PoolDBOutputService unavailable";
    return;
  }

  edm::ESTransientHandle<DDCompactView> pDD;
  edm::ESHandle<MuonDDDConstants> pMNDC;
  es.get<IdealGeometryRecord>().get(label_, pDD );
  es.get<MuonNumberingRecord>().get( pMNDC );

  const DDCompactView& cpv = *pDD;
  DTGeometryParsFromDD dtgp;

  dtgp.build( &cpv, *pMNDC, *rig );

  if ( mydbservice->isNewTagRequest("DTRecoGeometryRcd") ) {
    mydbservice->createNewIOV<RecoIdealGeometry>(rig
                                                 , mydbservice->beginOfTime()
                                                 , mydbservice->endOfTime()
                                                 , "DTRecoGeometryRcd");
  } else {
    edm::LogError("DTRecoIdealDBLoader")<<"DTRecoGeometryRcd Tag is already present.";
  }
}
