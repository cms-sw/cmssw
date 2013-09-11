#include "GEMRecoIdealDBLoader.h"

#include "Geometry/GEMGeometryBuilder/src/GEMGeometryParsFromDD.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "Geometry/Records/interface/GEMRecoGeometryRcd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"

#include <iostream>

using namespace std;

GEMRecoIdealDBLoader::GEMRecoIdealDBLoader(const edm::ParameterSet& iConfig) : label_()
{
  std::cout<<"GEMRecoIdealDBLoader::GEMRecoIdealDBLoader"<<std::endl;
}

GEMRecoIdealDBLoader::~GEMRecoIdealDBLoader()
{
  std::cout<<"GEMRecoIdealDBLoader::~GEMRecoIdealDBLoader"<<std::endl;
}

void
GEMRecoIdealDBLoader::beginRun( const edm::Run&, edm::EventSetup const& es) 
{
  RecoIdealGeometry* rig = new RecoIdealGeometry;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    edm::LogError("GEMRecoIdealDBLoader")<<"PoolDBOutputService unavailable";
    return;
  }

  edm::ESTransientHandle<DDCompactView> pDD;
  edm::ESHandle<MuonDDDConstants> pMNDC;
  es.get<IdealGeometryRecord>().get(label_, pDD );
  es.get<MuonNumberingRecord>().get( pMNDC );

  const DDCompactView& cpv = *pDD;
  GEMGeometryParsFromDD rpcpd;

  rpcpd.build( &cpv, *pMNDC, *rig );

  if ( mydbservice->isNewTagRequest("GEMRecoGeometryRcd") ) {
    mydbservice->createNewIOV<RecoIdealGeometry>(rig
                                                 , mydbservice->beginOfTime()
                                                 , mydbservice->endOfTime()
                                                 , "GEMRecoGeometryRcd");
  } else {
    edm::LogError("GEMRecoIdealDBLoader")<<"GEMRecoGeometryRcd Tag is already present.";
  }
}
