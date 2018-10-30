#include "GeometricTimingDetLoader.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/GeometryObjects/interface/PGeometricTimingDet.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include <DetectorDescription/Core/interface/DDCompactView.h>
#include <DetectorDescription/Core/interface/DDExpandedView.h>

#include "Geometry/Records/interface/MTDGeometryRecord.h"

#include <iostream>
#include <string>
#include <vector>

// just a reminder to self... beware errors caused by levels.  Look
// at how tracker is built and how GeometricSearchTracker.h is built 
// up from the hierarchy.

GeometricTimingDetLoader::GeometricTimingDetLoader(const edm::ParameterSet& iConfig)
{
  LogDebug("GeometricTimingDetLoader") 
    <<"GeometricTimingDetLoader::GeometricTimingDetLoader"<<std::endl;
}

GeometricTimingDetLoader::~GeometricTimingDetLoader()
{
  LogDebug("GeometricTimingDetLoader") 
    <<"GeometricTimingDetLoader::~GeometricTimingDetLoader"<<std::endl;
}

void
GeometricTimingDetLoader::beginRun( edm::Run const& /* iEvent */, edm::EventSetup const& es) 
{
  std::cout<<"GeometricTimingDetLoader::beginJob"<<std::endl;
  PGeometricTimingDet* pgd = new PGeometricTimingDet;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    std::cout<<"PoolDBOutputService unavailable"<<std::endl;
    return;
  }
  edm::ESHandle<DDCompactView> pDD;
  edm::ESHandle<GeometricTimingDet> rDD;
  es.get<IdealGeometryRecord>().get( pDD );
  es.get<IdealGeometryRecord>().get( rDD );
  const GeometricTimingDet* tracker = &(*rDD);

  // so now I have the tracker itself. loop over all its components to store them.
  putOne(tracker, pgd, 0);
  std::vector<const GeometricTimingDet*> tc = tracker->components();
  std::cout <<"Tracker has " << tc.size() << " components." << std::endl; //, lets go through them." << std::endl;
  std::vector<const GeometricTimingDet*>::const_iterator git = tc.begin();
  std::vector<const GeometricTimingDet*>::const_iterator egit = tc.end();
  int count=0;
  int lev = 1;
  for (; git!= egit; ++git) {  // one level below "tracker"
    putOne(*git, pgd, lev);
    std::vector<const GeometricTimingDet*> inone = (*git)->components();
    //    << ctste.name((*git)->type())
    //    std::cout << lev << " type " << (*git)->type() << " " << int((*git)->geographicalId()) << std::endl; // << " has " << inone.size() << " components." << std::endl;
    if ( inone.empty() )  ++count;
    std::vector<const GeometricTimingDet*>::const_iterator git2 = inone.begin();
    std::vector<const GeometricTimingDet*>::const_iterator egit2 = inone.end();
    ++lev;
    for (; git2 != egit2; ++git2) { // level 2
      putOne(*git2, pgd, lev);
      std::vector<const GeometricTimingDet*> intwo= (*git2)->components();
      //      std::cout << lev << "\ttype " << (*git2)->type() << " " << int((*git2)->geographicalId()) << std::endl; // << " has " << intwo.size() << " components." << std::endl;
      if ( intwo.empty() )  ++count;
      std::vector<const GeometricTimingDet*>::const_iterator git3 = intwo.begin();
      std::vector<const GeometricTimingDet*>::const_iterator egit3 = intwo.end();
      ++lev;
      for (; git3 != egit3; ++git3) { // level 3
	putOne(*git3, pgd, lev);
	std::vector<const GeometricTimingDet*> inthree= (*git3)->components();
	//	std::cout << lev << "\t\ttype " << (*git3)->type() << " " << int((*git3)->geographicalId()) << std::endl; // << " has " << inthree.size() << " components." << std::endl;
	if ( inthree.empty() )  ++count;
	std::vector<const GeometricTimingDet*>::const_iterator git4 = inthree.begin();
	std::vector<const GeometricTimingDet*>::const_iterator egit4 = inthree.end();
	++lev;
	for (; git4 != egit4; ++git4) { //level 4
	  putOne(*git4, pgd, lev);
	  std::vector<const GeometricTimingDet*> infour= (*git4)->components();
	  //	  std::cout << lev << "\t\t\ttype " << (*git4)->type() << " " << int((*git4)->geographicalId()) << std::endl; // << " has " << infour.size() << " components." << std::endl;
	  if ( infour.empty() )  ++count;
	  std::vector<const GeometricTimingDet*>::const_iterator git5 = infour.begin();
	  std::vector<const GeometricTimingDet*>::const_iterator egit5 = infour.end();
	  ++lev;
	  for (; git5 != egit5; ++git5) { // level 5
	    putOne(*git5, pgd, lev);
	    std::vector<const GeometricTimingDet*> infive= (*git5)->components();
	    //	    std::cout << lev << "\t\t\t\ttype " << (*git5)->type() << " " << int((*git5)->geographicalId()) << std::endl; // << " has " << infive.size() << " components." << std::endl;
	    if ( infive.empty() )  ++count;
	    std::vector<const GeometricTimingDet*>::const_iterator git6 = infive.begin();
	    std::vector<const GeometricTimingDet*>::const_iterator egit6 = infive.end();
	    ++lev;
	    for (; git6 != egit6; ++git6) { //level 6
	      putOne(*git6, pgd, lev);
	      std::vector<const GeometricTimingDet*> insix= (*git6)->components();
	      //	      std::cout << lev << "\t\t\t\t\ttype " << (*git6)->type() << " " << int((*git6)->geographicalId()) << std::endl; // << " has " << insix.size() << " components." << std::endl;
	      if ( insix.empty() )  ++count;
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
  std::vector<const GeometricTimingDet*> modules =  tracker->deepComponents();
  std::cout << " No. of Tracker components \"deepComponents\" = " << modules.size() << std::endl;
  std::cout << " Counted # of lowest \"leaves\" = " << count << std::endl; 
  if ( mydbservice->isNewTagRequest("PGeometricTimingDetRcd") ) {
    mydbservice->createNewIOV<PGeometricTimingDet>( pgd
					      , mydbservice->beginOfTime()
					      , mydbservice->endOfTime()
					      , "PGeometricTimingDetRcd");
  } else {
    std::cout << "PGeometricTimingDetRcd Tag is already present." << std::endl;
  }
}
  
void GeometricTimingDetLoader::putOne ( const GeometricTimingDet* gd, PGeometricTimingDet* pgd, int lev ) {

//   std::cout << "putting name: " << gd->name().name();
//   std::cout << " gid: " << gd->geographicalID();
//   std::cout << " type: " << gd->type() << std::endl;
//  std::cout << "shape = " << gd->shape()<<"; name = "<<gd->name().name()<<"; parameter number = "<<gd->params().size()<<std::endl;
  PGeometricTimingDet::Item item;
  const DDTranslation& tran = gd->translation();
  const DDRotationMatrix& rot = gd->rotation();
  DD3Vector x, y, z;
  rot.GetComponents(x, y, z);
  item.name_           = gd->name().name();
  item.level_          = lev;
  item.x_              = tran.X();
  item.y_              = tran.Y();
  item.z_              = tran.Z();
  item.phi_            = gd->phi();
  item.rho_            = gd->rho();
  item.a11_            = x.X();
  item.a12_            = y.X();
  item.a13_            = z.X();
  item.a21_            = x.Y();
  item.a22_            = y.Y();
  item.a23_            = z.Y();
  item.a31_            = x.Z();
  item.a32_            = y.Z();
  item.a33_            = z.Z();
  item.shape_          = static_cast<int>(gd->shape());
  item.type_           = gd->type();
  if(gd->shape()==DDSolidShape::ddbox){
    item.params_0=gd->params()[0];
    item.params_1=gd->params()[1];
    item.params_2=gd->params()[2];
    item.params_3=0;
    item.params_4=0;
    item.params_5=0;
    item.params_6=0;
    item.params_7=0;
    item.params_8=0;
    item.params_9=0;
    item.params_10=0;
  }else if(gd->shape()==DDSolidShape::ddtrap){
    item.params_0=gd->params()[0];
    item.params_1=gd->params()[1];
    item.params_2=gd->params()[2];
    item.params_3=gd->params()[3];
    item.params_4=gd->params()[4];
    item.params_5=gd->params()[5];
    item.params_6=gd->params()[6];
    item.params_7=gd->params()[7];
    item.params_8=gd->params()[8];
    item.params_9=gd->params()[9];
    item.params_10=gd->params()[10];
  }else{
    item.params_0=0;
    item.params_1=0;
    item.params_2=0;
    item.params_3=0;
    item.params_4=0;
    item.params_5=0;
    item.params_6=0;
    item.params_7=0;
    item.params_8=0;
    item.params_9=0;
    item.params_10=0;
  } 
  item.geographicalID_ = gd->geographicalID();
  // FIXME: These are moved to PGeometricTimingDetExtra:
  //item.volume_         = gd->volume();
  //item.density_        = gd->density();
  //item.weight_         = gd->weight();
  //item.copy_           = gd->copyno();
  //item.material_       = gd->material();
  item.radLength_      = gd->radLength();
  item.xi_             = gd->xi();
  item.pixROCRows_     = gd->pixROCRows();
  item.pixROCCols_     = gd->pixROCCols();
  item.pixROCx_        = gd->pixROCx();
  item.pixROCy_        = gd->pixROCy();
  item.stereo_         = gd->stereo();
  item.siliconAPVNum_ = gd->siliconAPVNum();
  pgd->pgeomdets_.push_back ( item );
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GeometricTimingDetLoader);
