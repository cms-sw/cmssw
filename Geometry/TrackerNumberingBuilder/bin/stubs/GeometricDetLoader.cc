#include "GeometricDetLoader.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/IdealGeometryObjects/interface/PGeometricDet.h"
#include "CondFormats/DataRecord/interface/PGeometricDetRcd.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include <DetectorDescription/Core/interface/DDCompactView.h>

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>

GeometricDetLoader::GeometricDetLoader(const edm::ParameterSet& iConfig)
{
  std::cout<<"GeometricDetLoader::GeometricDetLoader"<<std::endl;
}

GeometricDetLoader::~GeometricDetLoader()
{
  std::cout<<"GeometricDetLoader::~GeometricDetLoader"<<std::endl;
}

void
GeometricDetLoader::beginJob( edm::EventSetup const& es) 
{
  std::cout<<"GeometricDetLoader::beginJob"<<std::endl;
  PGeometricDet* pgd = new PGeometricDet;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    std::cout<<"PoolDBOutputService unavailable"<<std::endl;
    return;
  }
  edm::ESHandle<DDCompactView> pDD;
  edm::ESHandle<GeometricDet> rDD;
  es.get<IdealGeometryRecord>().get( pDD );
  //  const DDCompactView& cpv = *pDD;
  es.get<IdealGeometryRecord>().get( rDD );
  //  DDDCmsTrackerContruction theDDDCmsTrackerContruction;
  //  const GeometricDet* tracker = theDDDCmsTrackerContruction.construct(&(*pDD));
  const GeometricDet* tracker = &(*rDD);
  // so now I have the tracker itself... I think this is what I want to do...
  putOne(tracker, pgd);
  std::vector<const GeometricDet*> modules =  tracker->deepComponents();
  for(unsigned int i=0; i<modules.size();i++){
    putOne(modules[i], pgd);
  }
  if ( mydbservice->isNewTagRequest("PGeometricDetRcd") ) {
    mydbservice->createNewIOV<PGeometricDet>(pgd, mydbservice->endOfTime(), "PGeometricDetRcd");
  } else {
    std::cout << "PGeometricDetRcd Tag is already present." << std::endl;
  }
}
  
void GeometricDetLoader::putOne ( const GeometricDet* gd, PGeometricDet* pgd ) {

  std::cout << "putting name: " << gd->name().name();
  std::cout << " gid: " << gd->geographicalID();
  std::cout << " type: " << gd->type() << std::endl;
  PGeometricDet::Item item;
  DDTranslation tran = gd->translation();
  DDRotationMatrix rot = gd->rotation();
  DD3Vector x, y, z;
  rot.GetComponents(x, y, z);
  item._name           = gd->name().name();
  item._x              = tran.X();
  item._y              = tran.Y();
  item._z              = tran.Z();
  item._phi            = gd->phi();
  item._rho            = gd->rho();
  item._a11            = x.X();
  item._a12            = y.X();
  item._a13            = z.X();
  item._a21            = x.Y();
  item._a22            = y.Y();
  item._a23            = z.Y();
  item._a31            = x.Z();
  item._a32            = y.Z();
  item._a33            = z.Z();
  item._shape          = gd->shape();
  item._type           = gd->type();
  item._params         = gd->params();
  item._geographicalID = gd->geographicalID();
  item._volume         = gd->volume();
  item._density        = gd->density();
  item._weight         = gd->weight();
  item._copy           = gd->copyno();
  item._material       = gd->material();
  item._radLength      = gd->radLength();
  item._xi             = gd->xi();
  item._pixROCRows     = gd->pixROCRows();
  item._pixROCCols     = gd->pixROCCols();
  item._pixROCx        = gd->pixROCx();
  item._pixROCy        = gd->pixROCy();
  item._stereo         = gd->stereo();
  item._siliconAPVNum = gd->siliconAPVNum();
  pgd->pgeomdets_.push_back ( item );
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GeometricDetLoader);
