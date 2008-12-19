#include "CSCRecoIdealDBLoader.h"

#include <Geometry/CSCGeometryBuilder/src/CSCGeometryParsFromDD.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "CondFormats/GeometryObjects/interface/CSCRecoDigiParameters.h"
#include "Geometry/Records/interface/CSCRecoGeometryRcd.h"
#include "Geometry/Records/interface/CSCRecoDigiParametersRcd.h"
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

CSCRecoIdealDBLoader::CSCRecoIdealDBLoader(const edm::ParameterSet& iConfig) : label_()
{
  std::cout<<"CSCRecoIdealDBLoader::CSCRecoIdealDBLoader"<<std::endl;
}

CSCRecoIdealDBLoader::~CSCRecoIdealDBLoader()
{
  std::cout<<"CSCRecoIdealDBLoader::~CSCRecoIdealDBLoader"<<std::endl;
}

void
CSCRecoIdealDBLoader::beginJob( edm::EventSetup const& es) 
{
  std::cout<<"CSCRecoIdealDBLoader::beginJob"<<std::endl;
  RecoIdealGeometry* rig = new RecoIdealGeometry;
  CSCRecoDigiParameters* rdp = new CSCRecoDigiParameters;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    std::cout<<"PoolDBOutputService unavailable"<<std::endl;
    return;
  }

  edm::ESHandle<DDCompactView> pDD;
  edm::ESHandle<MuonDDDConstants> pMNDC;
  es.get<IdealGeometryRecord>().get(label_, pDD );
  es.get<MuonNumberingRecord>().get( pMNDC );

  const DDCompactView& cpv = *pDD;
  CSCGeometryParsFromDD cscgp;

  cscgp.build( &cpv, *pMNDC, *rig, *rdp );

  if ( mydbservice->isNewTagRequest("CSCRecoGeometryRcd") ) {
    mydbservice->createNewIOV<RecoIdealGeometry>(rig
						 , mydbservice->beginOfTime()
						 , mydbservice->endOfTime()
						 , "CSCRecoGeometryRcd");
  } else {
    std::cout << "RecoIdealGeometryRcd Tag is already present." << std::endl;
  }
  if ( mydbservice->isNewTagRequest("CSCRecoDigiParametersRcd") ) {
    mydbservice->createNewIOV<CSCRecoDigiParameters>(rdp
						     , mydbservice->beginOfTime()
						     , mydbservice->endOfTime()
						     , "CSCRecoDigiParametersRcd");
  } else {
    std::cout << "CSCRecoDigiParametersRcd Tag is already present." << std::endl;
  }

}
