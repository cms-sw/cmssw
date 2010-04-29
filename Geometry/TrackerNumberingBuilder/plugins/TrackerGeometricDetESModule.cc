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
  std::cout << " Start produceGDE with gd.size = " << gd->deepComponents().size() << std::endl;
//   const  std::vector< GeometricDet const *>& godeep(gd->deepComponents());
  if (fromDDD_) {
    // iterate over the whole GeometricDet, i.e. do the first one then get deepComponents and iterate over those
    //something like edm::ESTransientHandle<DDCompactView> cpv & etc.... can it use/rely on the above produce? yes
    const GeometricDet* tracker = &(*gd);
    edm::ESTransientHandle<DDCompactView> cpv;
    iRecord.get( cpv );
    DDExpandedView ev(*cpv);
    ev.goTo(tracker->navType());
    putOne((*cpv), (*gde), tracker, ev, 0);
    std::vector<const GeometricDet*> tc = tracker->components();
    std::vector<const GeometricDet*>::const_iterator git = tc.begin();
    std::vector<const GeometricDet*>::const_iterator egit = tc.end();
    int count=0;
    int lev = 1;
    //  CmsTrackerStringToEnum ctste;
    for (; git!= egit; ++git) {  // one level below "tracker"
      ev.goTo((*git)->navType());
      putOne((*cpv), (*gde), *git, ev, lev);
      std::vector<const GeometricDet*> inone = (*git)->components();
      //    << ctste.name((*git)->type())
      //    std::cout << lev << " type " << (*git)->type() << " " << int((*git)->geographicalId()) << std::endl; // << " has " << inone.size() << " components." << std::endl;
      if ( inone.size() == 0 )  ++count;
      std::vector<const GeometricDet*>::const_iterator git2 = inone.begin();
      std::vector<const GeometricDet*>::const_iterator egit2 = inone.end();
      ++lev;
      for (; git2 != egit2; ++git2) { // level 2
	ev.goTo((*git2)->navType());
	putOne((*cpv), (*gde), *git2, ev, lev);
	std::vector<const GeometricDet*> intwo= (*git2)->components();
	//      std::cout << lev << "\ttype " << (*git2)->type() << " " << int((*git2)->geographicalId()) << std::endl; // << " has " << intwo.size() << " components." << std::endl;
	if ( intwo.size() == 0 )  ++count;
	std::vector<const GeometricDet*>::const_iterator git3 = intwo.begin();
	std::vector<const GeometricDet*>::const_iterator egit3 = intwo.end();
	++lev;
	for (; git3 != egit3; ++git3) { // level 3
	  ev.goTo((*git3)->navType());
	  putOne((*cpv), (*gde), *git3, ev, lev);
	  std::vector<const GeometricDet*> inthree= (*git3)->components();
	  //std::cout << lev << "\t\ttype " << (*git3)->type() << " " << int((*git3)->geographicalId()) << std::endl; // << " has " << inthree.size() << " components." << std::endl;
	  if ( inthree.size() == 0 )  ++count;
	  std::vector<const GeometricDet*>::const_iterator git4 = inthree.begin();
	  std::vector<const GeometricDet*>::const_iterator egit4 = inthree.end();
	  ++lev;
	  for (; git4 != egit4; ++git4) { //level 4
	    ev.goTo((*git4)->navType());
	    putOne((*cpv), (*gde), *git4, ev, lev);
	    std::vector<const GeometricDet*> infour= (*git4)->components();
	    //  std::cout << lev << "\t\t\ttype " << (*git4)->type() << " " << int((*git4)->geographicalId()) << std::endl; // << " has " << infour.size() << " components." << std::endl;
	    if ( infour.size() == 0 )  ++count;
	    std::vector<const GeometricDet*>::const_iterator git5 = infour.begin();
	    std::vector<const GeometricDet*>::const_iterator egit5 = infour.end();
	    ++lev;
	    for (; git5 != egit5; ++git5) { // level 5
	      ev.goTo((*git5)->navType());
	      putOne((*cpv), (*gde), *git5, ev, lev);
	      std::vector<const GeometricDet*> infive= (*git5)->components();
	      //    std::cout << lev << "\t\t\t\ttype " << (*git5)->type() << " " << int((*git5)->geographicalId()) << std::endl; // << " has " << infive.size() << " components." << std::endl;
	      if ( infive.size() == 0 )  ++count;
	      std::vector<const GeometricDet*>::const_iterator git6 = infive.begin();
	      std::vector<const GeometricDet*>::const_iterator egit6 = infive.end();
	      ++lev;
	      for (; git6 != egit6; ++git6) { //level 6
		ev.goTo((*git6)->navType());
		putOne((*cpv), (*gde), *git6, ev, lev);
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
//     gde->reserve(godeep.size());

//     DDExpandedView ev(*cpv);
//     ev.goTo(gd->navType());
//     // do the "root" tracker first.
//     std::string matname = ((ev.logicalPart()).material()).name().fullname();
//     std::vector<DDExpandedNode> evs(GeometricDetExtra::GeoHistory(ev.geoHistory().begin(),ev.geoHistory().end()));
//     std::cout << "Id " << gd->geographicalId() << " at address " << &(*gd) << std::endl;
//     const GeometricDet* const ge(&(*gd));
//     std::cout << " ge = " << ge << std::endl;
//     gde->push_back(GeometricDetExtra( ge, gd->geographicalId(), evs,
// 				      ((ev.logicalPart()).solid()).volume(), ((ev.logicalPart()).material()).density(), 
// 				      ((ev.logicalPart()).material()).density() * ( ((ev.logicalPart()).solid()).volume() / 1000.),
// 				      ev.copyno(), matname, true ));
//      //NOT USED (ev.logicalPart()).weight();
//     // then do the rest
//     std::vector< GeometricDet const *>::const_iterator git (godeep.begin()), gEndit(godeep.end());
//     for (; git != gEndit; ++git) {
//       ev.goTo((*git)->navType());
//       matname = ((ev.logicalPart()).material()).name().fullname();
//       evs = GeometricDetExtra::GeoHistory(ev.geoHistory().begin(),ev.geoHistory().end());
//       GeometricDet const * ge1(&(**git));
//       std::cout << " ge1 = " << ge1 << std::endl;
//       std::cout << "Id " << (*git)->geographicalId() << " at address " << &(**git) << std::endl;
//       gde->push_back(GeometricDetExtra( &(**git), (*git)->geographicalId(), evs,  
// 				      ((ev.logicalPart()).solid()).volume(), ((ev.logicalPart()).material()).density(), 
// 				      ((ev.logicalPart()).material()).density() * ( ((ev.logicalPart()).solid()).volume() / 1000.),
// 				      ev.copyno(), matname, true ));
//     }
//     // edm::ESHandle<GeometricDet> gd & etc;
//     // gde->reserve(gd.deepComponents().size());
//     //constructfromDD(cpv, gd, gde); //pass in by reference.
  }else{
    //edm::ESHandle<PGeometricDetExtra> pgde;
    //iRecord.get( pgde );
    //CondDBGDEConstructor c;
    //    gde->reserve(pgde->pGDExtra.size());
    //    c.construct( pgde, gde );
  }
  //   std::cout << " Start produceGDE with gdEXTRA.size = " << gde->size() << std::endl;
  return boost::shared_ptr<std::vector<GeometricDetExtra> >(gde);
}

void TrackerGeometricDetESModule::putOne(const DDCompactView& cpv, std::vector<GeometricDetExtra> & gde, const GeometricDet* gd, const DDExpandedView& ev, int lev ) {
  std::string matname = ((ev.logicalPart()).material()).name().fullname();
  std::vector<DDExpandedNode> evs = GeometricDetExtra::GeoHistory(ev.geoHistory().begin(),ev.geoHistory().end());
  std::cout << "name " << ev.logicalPart().name().name() << std::endl;
  if (ev.logicalPart().name().name() == "PixelBarrelActiveFull") {
    std::cout << "vol " << ev.logicalPart().solid().volume() << " dens " << ev.logicalPart().material().density() << std::endl;
    //      std::cout << "vol " << volume << " dens " << density << std::endl;
    // FOR DEBUG
    // I want to find this node in the filtered view and dump it out... can I? 
    std::string attribute = "TkDDDStructure"; // could come from .orcarc
    std::string value     = "any";
    DDSpecificsFilter filter;
    DDValue ddv(attribute,value,0);
    filter.setCriteria(ddv,DDSpecificsFilter::not_equals);    
    DDFilteredView fv(cpv); 
    fv.addFilter(filter);
    CmsTrackerStringToEnum theCmsTrackerStringToEnum;
    if (theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(attribute,&fv)) != GeometricDet::Tracker){
      fv.firstChild();
      if (theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(attribute,&fv)) != GeometricDet::Tracker){
	throw cms::Exception("Configuration") <<" The first child of the DDFilteredView is not what is expected \n"
					      <<ExtractStringFromDDD::getString(attribute,&fv)<<"\n";
      }
    }
    do {
      if (!fv.next()) break;
    } while (fv.navPos() != ev.navPos());
    std::cout <<  "fv vol " << fv.logicalPart().solid().volume() << " dens " << fv.logicalPart().material().density() << std::endl;
  }
  //  GeometricDet const * ge1(gd);
//   std::cout << " ge1 = " << ge1 << std::endl;
//   std::cout << "Id " << gd->geographicalId() << " at address " << gd << std::endl;
//   double d = (ev.logicalPart().material()).density();
//   double v = (ev.logicalPart().solid()).volume();
//   double w = v*d/100;
//   std::cout << "logical Part: " << ev.logicalPart() << std::endl;
//   std::cout << "solid: " << ev.logicalPart().solid() << std::endl;
//   std::cout << "material " << ev.logicalPart().material() << std::endl;
//   std::cout << "density d " << d << " * volume v " << v << " = weight w " << w << std::endl;
//   std::cout << ev.logicalPart().name().ns() << ":" << ev.logicalPart().name().name()<<std::endl;
//   DDLogicalPart lp = ev.logicalPart();

//   if (!lp) std::cout << "Oh SHIT!" << std::endl; 
//   DDMaterial mat= lp.material();
//   DDSolid sol= lp.solid(); //(DDName(lp.solid().name().name(), lp.solid().name().ns()));
//   DDPolycone poly(sol);
//   std::cout << DDSolidShapesName::name(sol.shape()) << " ";
//   std::cout << " density = " << mat.density() << " ";
//   std::cout << " volume = " << sol.volume() << " "; 
//   std::cout << " volume = " << poly.volume() << std::endl;
//   exit(0);
  gde.push_back(GeometricDetExtra( gd, gd->geographicalId(), evs,
				   ((ev.logicalPart()).solid()).volume(), ((ev.logicalPart()).material()).density(),
				   ((ev.logicalPart()).material()).density() * ( ((ev.logicalPart()).solid()).volume() / 1000.),                                                                       
				   ev.copyno(), matname, true ));
  if (ev.logicalPart().name().name() == "PixelBarrelActiveFull") {
    std::cout << "vol " << gde.back().volume() << " dens " << gde.back().density() << std::endl;
    //    std::cout << "vol " << volume << " dens " << density << std::endl;
  }

}

void TrackerGeometricDetESModule::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
						 const edm::IOVSyncValue & iosv, 
						 edm::ValidityInterval & oValidity)
{
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerGeometricDetESModule);
