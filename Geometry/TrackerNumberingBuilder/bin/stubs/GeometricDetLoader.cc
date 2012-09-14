#include "GeometricDetLoader.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include <DetectorDescription/Core/interface/DDCompactView.h>
#include <DetectorDescription/Core/interface/DDExpandedView.h>

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include <iostream>
#include <string>
#include <vector>
//#include <map>
//#include <sstream>
//#include <algorithm>

// just a reminder to self... beware errors caused by levels.  Look
// at how tracker is built and how GeometricSearchTracker.h is built 
// up from the hierarchy.

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
  es.get<IdealGeometryRecord>().get( rDD );
  const GeometricDet* tracker = &(*rDD);

  // so now I have the tracker itself. loop over all its components to store them.
  putOne(tracker, pgd, 0);
  std::vector<const GeometricDet*> tc = tracker->components();
  std::cout <<"Tracker has " << tc.size() << " components." << std::endl; //, lets go through them." << std::endl;
  std::vector<const GeometricDet*>::const_iterator git = tc.begin();
  std::vector<const GeometricDet*>::const_iterator egit = tc.end();
  int count=0;
  int lev = 1;
  for (; git!= egit; ++git) {  // one level below "tracker"
    putOne(*git, pgd, lev);
    std::vector<const GeometricDet*> inone = (*git)->components();
    //    << ctste.name((*git)->type())
    //    std::cout << lev << " type " << (*git)->type() << " " << int((*git)->geographicalId()) << std::endl; // << " has " << inone.size() << " components." << std::endl;
    if ( inone.size() == 0 )  ++count;
    std::vector<const GeometricDet*>::const_iterator git2 = inone.begin();
    std::vector<const GeometricDet*>::const_iterator egit2 = inone.end();
    ++lev;
    for (; git2 != egit2; ++git2) { // level 2
      putOne(*git2, pgd, lev);
      std::vector<const GeometricDet*> intwo= (*git2)->components();
      //      std::cout << lev << "\ttype " << (*git2)->type() << " " << int((*git2)->geographicalId()) << std::endl; // << " has " << intwo.size() << " components." << std::endl;
      if ( intwo.size() == 0 )  ++count;
      std::vector<const GeometricDet*>::const_iterator git3 = intwo.begin();
      std::vector<const GeometricDet*>::const_iterator egit3 = intwo.end();
      ++lev;
      for (; git3 != egit3; ++git3) { // level 3
	putOne(*git3, pgd, lev);
	std::vector<const GeometricDet*> inthree= (*git3)->components();
	//	std::cout << lev << "\t\ttype " << (*git3)->type() << " " << int((*git3)->geographicalId()) << std::endl; // << " has " << inthree.size() << " components." << std::endl;
	if ( inthree.size() == 0 )  ++count;
	std::vector<const GeometricDet*>::const_iterator git4 = inthree.begin();
	std::vector<const GeometricDet*>::const_iterator egit4 = inthree.end();
	++lev;
	for (; git4 != egit4; ++git4) { //level 4
	  putOne(*git4, pgd, lev);
	  std::vector<const GeometricDet*> infour= (*git4)->components();
	  //	  std::cout << lev << "\t\t\ttype " << (*git4)->type() << " " << int((*git4)->geographicalId()) << std::endl; // << " has " << infour.size() << " components." << std::endl;
	  if ( infour.size() == 0 )  ++count;
	  std::vector<const GeometricDet*>::const_iterator git5 = infour.begin();
	  std::vector<const GeometricDet*>::const_iterator egit5 = infour.end();
	  ++lev;
	  for (; git5 != egit5; ++git5) { // level 5
	    putOne(*git5, pgd, lev);
	    std::vector<const GeometricDet*> infive= (*git5)->components();
	    //	    std::cout << lev << "\t\t\t\ttype " << (*git5)->type() << " " << int((*git5)->geographicalId()) << std::endl; // << " has " << infive.size() << " components." << std::endl;
	    if ( infive.size() == 0 )  ++count;
	    std::vector<const GeometricDet*>::const_iterator git6 = infive.begin();
	    std::vector<const GeometricDet*>::const_iterator egit6 = infive.end();
	    ++lev;
	    for (; git6 != egit6; ++git6) { //level 6
	      putOne(*git6, pgd, lev);
	      std::vector<const GeometricDet*> insix= (*git6)->components();
	      //	      std::cout << lev << "\t\t\t\t\ttype " << (*git6)->type() << " " << int((*git6)->geographicalId()) << std::endl; // << " has " << insix.size() << " components." << std::endl;
	      if ( insix.size() == 0 )  ++count;
	    } // level 6
	    --lev;
	  } // level 5
	  --lev;
	} // level 4
	--lev;
      } //level 3
      --lev;
    } // level 2
    --lev;
  }
  std::vector<const GeometricDet*> modules =  tracker->deepComponents();
  std::cout << " No. of Tracker components \"deepComponents\" = " << modules.size() << std::endl;
  std::cout << " Counted # of lowest \"leaves\" = " << count << std::endl; 
  if ( mydbservice->isNewTagRequest("PGeometricDetRcd") ) {
    mydbservice->createNewIOV<PGeometricDet>( pgd
					      , mydbservice->beginOfTime()
					      , mydbservice->endOfTime()
					      , "PGeometricDetRcd");
  } else {
    std::cout << "PGeometricDetRcd Tag is already present." << std::endl;
  }
}
  
void GeometricDetLoader::putOne ( const GeometricDet* gd, PGeometricDet* pgd, int lev ) {

//   std::cout << "putting name: " << gd->name().name();
//   std::cout << " gid: " << gd->geographicalID();
//   std::cout << " type: " << gd->type() << std::endl;
//  std::cout << "shape = " << gd->shape()<<"; name = "<<gd->name().name()<<"; parameter number = "<<gd->params().size()<<std::endl;
  PGeometricDet::Item item;
  DDTranslation tran = gd->translation();
  DDRotationMatrix rot = gd->rotation();
  DD3Vector x, y, z;
  rot.GetComponents(x, y, z);
  item._name           = gd->name().name();
  item._level          = lev;
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
  if(gd->shape()==1){
    item._params0=gd->params()[0];
    item._params1=gd->params()[1];
    item._params2=gd->params()[2];
    item._params3=0;
    item._params4=0;
    item._params5=0;
    item._params6=0;
    item._params7=0;
    item._params8=0;
    item._params9=0;
    item._params10=0;
  }else if(gd->shape()==3){
    item._params0=gd->params()[0];
    item._params1=gd->params()[1];
    item._params2=gd->params()[2];
    item._params3=gd->params()[3];
    item._params4=gd->params()[4];
    item._params5=gd->params()[5];
    item._params6=gd->params()[6];
    item._params7=gd->params()[7];
    item._params8=gd->params()[8];
    item._params9=gd->params()[9];
    item._params10=gd->params()[10];
  }else{
    item._params0=0;
    item._params1=0;
    item._params2=0;
    item._params3=0;
    item._params4=0;
    item._params5=0;
    item._params6=0;
    item._params7=0;
    item._params8=0;
    item._params9=0;
    item._params10=0;
  } 
  item._geographicalID = gd->geographicalID();
  // FIXME: These are moved to PGeometricDetExtra:
  //item._volume         = gd->volume();
  //item._density        = gd->density();
  //item._weight         = gd->weight();
  //item._copy           = gd->copyno();
  //item._material       = gd->material();
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
