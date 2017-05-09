#include "Geometry/TrackerNumberingBuilder/plugins/TrackerGeometricDetExtraESModule.h"
#include "Geometry/TrackerNumberingBuilder/plugins/DDDCmsTrackerContruction.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDetExtra.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PGeometricDetExtraRcd.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "ExtractStringFromDDD.h"
#include "CondDBCmsTrackerConstruction.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"


#include <memory>

using namespace edm;

TrackerGeometricDetExtraESModule::TrackerGeometricDetExtraESModule(const edm::ParameterSet & p) 
  : fromDDD_(p.getParameter<bool>("fromDDD")) 
{
  setWhatProduced(this);
}

TrackerGeometricDetExtraESModule::~TrackerGeometricDetExtraESModule() {}

boost::shared_ptr<std::vector<GeometricDetExtra> >
TrackerGeometricDetExtraESModule::produce(const IdealGeometryRecord & iRecord) {
  boost::shared_ptr<std::vector<GeometricDetExtra> > gde (new std::vector<GeometricDetExtra>);
  // get the GeometricDet which has a nav_type
  edm::ESHandle<GeometricDet> gd;
  iRecord.get ( gd );
  if (fromDDD_) {
  // traverse all components from the tracker down;
  // read the DD if from DD
    const GeometricDet* tracker = &(*gd);
    edm::ESTransientHandle<DDCompactView> cpv;
    iRecord.get( cpv );
    DDExpandedView ev(*cpv);
    ev.goTo(tracker->navType());
    putOne((*gde), tracker, ev, 0);
    auto tc = tracker->components();
    auto git = tc.begin();
    auto egit = tc.end();
    int count=0;
    int lev = 1;
    //  CmsTrackerStringToEnum ctst
    gde->reserve(tracker->deepComponents().size());
    for (; git!= egit; ++git) {  // one level below "tracker"
      ev.goTo((*git)->navType());
      putOne((*gde), *git, ev, lev);
      auto inone = (*git)->components();
      //    std::cout << lev << " type " << (*git)->type() << " " << int((*git)->geographicalId()) << std::endl; // << " has " << inone.size() << " components." << std::endl;
      if ( inone.size() == 0 )  ++count;
      auto git2 = inone.begin();
      auto egit2 = inone.end();
      ++lev;
      for (; git2 != egit2; ++git2) { // level 2
	ev.goTo((*git2)->navType());
	putOne((*gde), *git2, ev, lev);
	auto intwo= (*git2)->components();
	//      std::cout << lev << "\ttype " << (*git2)->type() << " " << int((*git2)->geographicalId()) << std::endl; // << " has " << intwo.size() << " components." << std::endl;
	if ( intwo.size() == 0 )  ++count;
	auto git3 = intwo.begin();
	auto egit3 = intwo.end();
	++lev;
	for (; git3 != egit3; ++git3) { // level 3
	  ev.goTo((*git3)->navType());
	  putOne((*gde), *git3, ev, lev);
	  auto inthree= (*git3)->components();
	  //std::cout << lev << "\t\ttype " << (*git3)->type() << " " << int((*git3)->geographicalId()) << std::endl; // << " has " << inthree.size() << " components." << std::endl;
	  if ( inthree.size() == 0 )  ++count;
	  auto git4 = inthree.begin();
	  auto egit4 = inthree.end();
	  ++lev;
	  for (; git4 != egit4; ++git4) { //level 4
	    ev.goTo((*git4)->navType());
	    putOne((*gde), *git4, ev, lev);
	    auto infour= (*git4)->components();
	    //  std::cout << lev << "\t\t\ttype " << (*git4)->type() << " " << int((*git4)->geographicalId()) << std::endl; // << " has " << infour.size() << " components." << std::endl;
	    if ( infour.size() == 0 )  ++count;
	    auto git5 = infour.begin();
	    auto egit5 = infour.end();
	    ++lev;
	    for (; git5 != egit5; ++git5) { // level 5
	      ev.goTo((*git5)->navType());
	      putOne((*gde), *git5, ev, lev);
	      auto infive= (*git5)->components();
	      //    std::cout << lev << "\t\t\t\ttype " << (*git5)->type() << " " << int((*git5)->geographicalId()) << std::endl; // << " has " << infive.size() << " components." << std::endl;
	      if ( infive.size() == 0 )  ++count;
	      auto git6 = infive.begin();
	      auto egit6 = infive.end();
	      ++lev;
	      for (; git6 != egit6; ++git6) { //level 6
		ev.goTo((*git6)->navType());
		putOne((*gde), *git6, ev, lev);
		auto insix= (*git6)->components();
		//      std::cout << lev << "\t\t\t\t\ttype " << (*git6)->type() << " " << int((*git6)->geographicalId()) << std::endl; // << " has " << insix.size() << " components." << std::endl;
		if ( insix.size() == 0 ){
		  ++count;
		} else {
		  edm::LogError("GeometricDetExtra") << "Hierarchy has exceeded hard-coded level 6 for Tracker " ;
		}
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
  }else{
    // if it is not from the DD, then just get the GDE from ES and match w/ GD.
    edm::ESHandle<PGeometricDetExtra> pgde;
    iRecord.getRecord<PGeometricDetExtraRcd>().get(pgde);
    std::map<uint32_t, const GeometricDet*> helperMap;
    const GeometricDet* tracker = &(*gd);
    helperMap[gd->geographicalID()] = tracker;
    auto tc = tracker->components();
    auto git = tc.begin();
    auto egit = tc.end();
    for (; git!= egit; ++git) {  // one level below "tracker"
      helperMap[(*git)->geographicalID()] = (*git);
      auto inone = (*git)->components();
      auto git2 = inone.begin();
      auto egit2 = inone.end();
      for (; git2 != egit2; ++git2) { // level 2
	helperMap[(*git2)->geographicalID()] = (*git2);
	auto intwo= (*git2)->components();
	auto git3 = intwo.begin();
	auto egit3 = intwo.end();
	for (; git3 != egit3; ++git3) { // level 3
	  helperMap[(*git3)->geographicalID()] = (*git3);
	  auto inthree= (*git3)->components();
	  auto git4 = inthree.begin();
	  auto egit4 = inthree.end();
	  for (; git4 != egit4; ++git4) { //level 4
	    helperMap[(*git4)->geographicalID()] = (*git4);
	    auto infour= (*git4)->components();
	    auto git5 = infour.begin();
	    auto egit5 = infour.end();
	    for (; git5 != egit5; ++git5) { // level 5
	      helperMap[(*git5)->geographicalID()] = (*git5);
	      auto infive= (*git5)->components();
	      auto git6 = infive.begin();
	      auto egit6 = infive.end();
	      for (; git6 != egit6; ++git6) { //level 6
		helperMap[(*git6)->geographicalID()] = (*git6);
		if ( (*git6)->components().size() != 0 ){
		  edm::LogError("GeometricDetExtra") << "Hierarchy has exceeded hard-coded level of 6 for Tracker " ;
		}
	      } // level 6
	    } // level 5
	  } // level 4
	} //level 3
      } // level 2
    }
  
    const std::vector<PGeometricDetExtra::Item>& pgdes = pgde->pgdes_;
    gde->reserve(pgdes.size());
    std::vector<DDExpandedNode> evs; //EMPTY
    std::string nm; //EMPTY
    for (unsigned int i = 0; i < pgdes.size(); ++i) {
	//   GeometricDetExtra( GeometricDet const *gd, DetId id, GeoHistory& gh,  double vol, double dens, double wgt, double cpy, const std::string& mat, const std::string& name, bool dd=false );
      gde->push_back( GeometricDetExtra(helperMap[pgdes[i]._geographicalId], pgdes[i]._geographicalId, evs
				       , pgdes[i]._volume, pgdes[i]._density, pgdes[i]._weight, pgdes[i]._copy
				       , pgdes[i]._material, nm));
    }
  }
  return boost::shared_ptr<std::vector<GeometricDetExtra> >(gde);
}

void TrackerGeometricDetExtraESModule::putOne(std::vector<GeometricDetExtra> & gde, const GeometricDet* gd, const DDExpandedView& ev, int lev ) {
  std::string matname = ((ev.logicalPart()).material()).name().fullname();
  std::string lpname = ((ev.logicalPart()).name().fullname());
  std::vector<DDExpandedNode> evs = GeometricDetExtra::GeoHistory(ev.geoHistory().begin(),ev.geoHistory().end());
  gde.push_back(GeometricDetExtra( gd, gd->geographicalId(), evs,
				   ((ev.logicalPart()).solid()).volume(), ((ev.logicalPart()).material()).density(),
				   ((ev.logicalPart()).material()).density() * ( ((ev.logicalPart()).solid()).volume() / 1000.),                                                                       
				   ev.copyno(), matname, lpname, true ));
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerGeometricDetExtraESModule);
