#include "Geometry/TrackerNumberingBuilder/plugins/TrackerGeometricDetESModule.h"
#include "Geometry/TrackerNumberingBuilder/plugins/DDDCmsTrackerContruction.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
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

TrackerGeometricDetESModule::TrackerGeometricDetESModule(const edm::ParameterSet & p) 
  : fromDDD_(p.getParameter<bool>("fromDDD")) 
{
  setWhatProduced(this);
  setWhatProduced(this, &TrackerGeometricDetESModule::produceGDE);
}

TrackerGeometricDetESModule::~TrackerGeometricDetESModule() {}

std::auto_ptr<GeometricDet> 
TrackerGeometricDetESModule::produce(const IdealGeometryRecord & iRecord){ 
  if(fromDDD_){

    edm::ESTransientHandle<DDCompactView> cpv;
    iRecord.get( cpv );
    
    DDDCmsTrackerContruction theDDDCmsTrackerContruction;
    return std::auto_ptr<GeometricDet> (const_cast<GeometricDet*>(theDDDCmsTrackerContruction.construct(&(*cpv))));

  }else{

    edm::ESHandle<PGeometricDet> pgd;
    iRecord.get( pgd );
    
    CondDBCmsTrackerConstruction cdbtc;
    return std::auto_ptr<GeometricDet> ( const_cast<GeometricDet*>(cdbtc.construct( *pgd )) );
  }
}

boost::shared_ptr<std::vector<GeometricDetExtra> >
TrackerGeometricDetESModule::produceGDE(const IdealGeometryRecord & iRecord) {
  boost::shared_ptr<std::vector<GeometricDetExtra> > gde (new std::vector<GeometricDetExtra>);
  /// MEC for phase one, use original sources to build the new class.
  // this gets the one above :)
  edm::ESHandle<GeometricDet> gd;
  iRecord.get ( gd );
  if (fromDDD_) {
    const GeometricDet* tracker = &(*gd);
    edm::ESTransientHandle<DDCompactView> cpv;
    iRecord.get( cpv );
    DDExpandedView ev(*cpv);
    ev.goTo(tracker->navType());
    putOne((*gde), tracker, ev, 0);
    std::vector<const GeometricDet*> tc = tracker->components();
    std::vector<const GeometricDet*>::const_iterator git = tc.begin();
    std::vector<const GeometricDet*>::const_iterator egit = tc.end();
    int count=0;
    int lev = 1;
    //  CmsTrackerStringToEnum ctst
    gde->reserve(tracker->deepComponents().size());
    for (; git!= egit; ++git) {  // one level below "tracker"
      ev.goTo((*git)->navType());
      putOne((*gde), *git, ev, lev);
      std::vector<const GeometricDet*> inone = (*git)->components();
      //    << ctste.name((*git)->type())
      //    std::cout << lev << " type " << (*git)->type() << " " << int((*git)->geographicalId()) << std::endl; // << " has " << inone.size() << " components." << std::endl;
      if ( inone.size() == 0 )  ++count;
      std::vector<const GeometricDet*>::const_iterator git2 = inone.begin();
      std::vector<const GeometricDet*>::const_iterator egit2 = inone.end();
      ++lev;
      for (; git2 != egit2; ++git2) { // level 2
	ev.goTo((*git2)->navType());
	putOne((*gde), *git2, ev, lev);
	std::vector<const GeometricDet*> intwo= (*git2)->components();
	//      std::cout << lev << "\ttype " << (*git2)->type() << " " << int((*git2)->geographicalId()) << std::endl; // << " has " << intwo.size() << " components." << std::endl;
	if ( intwo.size() == 0 )  ++count;
	std::vector<const GeometricDet*>::const_iterator git3 = intwo.begin();
	std::vector<const GeometricDet*>::const_iterator egit3 = intwo.end();
	++lev;
	for (; git3 != egit3; ++git3) { // level 3
	  ev.goTo((*git3)->navType());
	  putOne((*gde), *git3, ev, lev);
	  std::vector<const GeometricDet*> inthree= (*git3)->components();
	  //std::cout << lev << "\t\ttype " << (*git3)->type() << " " << int((*git3)->geographicalId()) << std::endl; // << " has " << inthree.size() << " components." << std::endl;
	  if ( inthree.size() == 0 )  ++count;
	  std::vector<const GeometricDet*>::const_iterator git4 = inthree.begin();
	  std::vector<const GeometricDet*>::const_iterator egit4 = inthree.end();
	  ++lev;
	  for (; git4 != egit4; ++git4) { //level 4
	    ev.goTo((*git4)->navType());
	    putOne((*gde), *git4, ev, lev);
	    std::vector<const GeometricDet*> infour= (*git4)->components();
	    //  std::cout << lev << "\t\t\ttype " << (*git4)->type() << " " << int((*git4)->geographicalId()) << std::endl; // << " has " << infour.size() << " components." << std::endl;
	    if ( infour.size() == 0 )  ++count;
	    std::vector<const GeometricDet*>::const_iterator git5 = infour.begin();
	    std::vector<const GeometricDet*>::const_iterator egit5 = infour.end();
	    ++lev;
	    for (; git5 != egit5; ++git5) { // level 5
	      ev.goTo((*git5)->navType());
	      putOne((*gde), *git5, ev, lev);
	      std::vector<const GeometricDet*> infive= (*git5)->components();
	      //    std::cout << lev << "\t\t\t\ttype " << (*git5)->type() << " " << int((*git5)->geographicalId()) << std::endl; // << " has " << infive.size() << " components." << std::endl;
	      if ( infive.size() == 0 )  ++count;
	      std::vector<const GeometricDet*>::const_iterator git6 = infive.begin();
	      std::vector<const GeometricDet*>::const_iterator egit6 = infive.end();
	      ++lev;
	      for (; git6 != egit6; ++git6) { //level 6
		ev.goTo((*git6)->navType());
		putOne((*gde), *git6, ev, lev);
		std::vector<const GeometricDet*> insix= (*git6)->components();
		//      std::cout << lev << "\t\t\t\t\ttype " << (*git6)->type() << " " << int((*git6)->geographicalId()) << std::endl; // << " has " << insix.size() << " components." << std::endl;
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
  }else{
    //NOT YET WRITTEN: MEC, April 2010
     //edm::ESHandle<PGeometricDetExtra> pgde;
    //iRecord.get( pgde );
    //CondDBGDEConstructor c;
    //    gde->reserve(pgde->pGDExtra.size());
    //    c.construct( pgde, gde );
  }
  //   std::cout << " Start produceGDE with gdEXTRA.size = " << gde->size() << std::endl;
  return boost::shared_ptr<std::vector<GeometricDetExtra> >(gde);
}

void TrackerGeometricDetESModule::putOne(std::vector<GeometricDetExtra> & gde, const GeometricDet* gd, const DDExpandedView& ev, int lev ) {
  std::string matname = ((ev.logicalPart()).material()).name().fullname();
  std::vector<DDExpandedNode> evs = GeometricDetExtra::GeoHistory(ev.geoHistory().begin(),ev.geoHistory().end());
  gde.push_back(GeometricDetExtra( gd, gd->geographicalId(), evs,
				   ((ev.logicalPart()).solid()).volume(), ((ev.logicalPart()).material()).density(),
				   ((ev.logicalPart()).material()).density() * ( ((ev.logicalPart()).solid()).volume() / 1000.),                                                                       
				   ev.copyno(), matname, ev.logicalPart().name().name(), true ));
}

void TrackerGeometricDetESModule::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
						 const edm::IOVSyncValue & iosv, 
						 edm::ValidityInterval & oValidity)
{
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerGeometricDetESModule);
