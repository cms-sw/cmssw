#include "Geometry/MTDNumberingBuilder/plugins/MTDGeometricTimingDetExtraESModule.h"
#include "Geometry/MTDNumberingBuilder/plugins/DDDCmsMTDConstruction.h"
#include "CondFormats/GeometryObjects/interface/PGeometricTimingDet.h"
#include "CondFormats/GeometryObjects/interface/PGeometricTimingDetExtra.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PGeometricTimingDetExtraRcd.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "ExtractStringFromDDD.h"
#include "CondDBCmsMTDConstruction.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"


#include <memory>

using namespace edm;

MTDGeometricTimingDetExtraESModule::MTDGeometricTimingDetExtraESModule(const edm::ParameterSet & p) 
  : fromDDD_(p.getParameter<bool>("fromDDD")) 
{
  setWhatProduced(this);
}

MTDGeometricTimingDetExtraESModule::~MTDGeometricTimingDetExtraESModule() {}

std::unique_ptr<std::vector<GeometricTimingDetExtra> >
MTDGeometricTimingDetExtraESModule::produce(const IdealGeometryRecord & iRecord) {
  auto gde = std::make_unique<std::vector<GeometricTimingDetExtra> >();
  // get the GeometricTimingDet which has a nav_type
  edm::ESHandle<GeometricTimingDet> gd;
  iRecord.get ( gd );
  if (fromDDD_) {
  // traverse all components from the tracker down;
  // read the DD if from DD
    const GeometricTimingDet* tracker = &(*gd);
    edm::ESTransientHandle<DDCompactView> cpv;
    iRecord.get( cpv );
    DDExpandedView ev(*cpv);
    ev.goTo(tracker->navType());
    putOne((*gde), tracker, ev, 0);
    std::vector<const GeometricTimingDet*> tc = tracker->components();    
    int count=0;
    int lev = 1;
    //  CmsMTDStringToEnum ctst
    gde->reserve(tracker->deepComponents().size());
    for( const auto* git : tc ) {
      ev.goTo(git->navType());
      putOne((*gde), git, ev, lev);
      std::vector<const GeometricTimingDet*> inone = git->components();
      if ( inone.empty() )  ++count;
      ++lev;
      for( const auto* git2 : inone ) {
	ev.goTo(git2->navType());
	putOne((*gde), git2, ev, lev);
	std::vector<const GeometricTimingDet*> intwo= git2->components();
	if ( intwo.empty() )  ++count;
	++lev;
	for( const auto* git3 : intwo ) {
	  ev.goTo(git3->navType());
	  putOne((*gde), git3, ev, lev);
	  std::vector<const GeometricTimingDet*> inthree= git3->components();
	  if ( inthree.empty() )  ++count;
	  ++lev;
	  for( const auto* git4 : inthree ) {
	    ev.goTo(git4->navType());
	    putOne((*gde), git4, ev, lev);
	    std::vector<const GeometricTimingDet*> infour= git4->components();
	    if ( infour.empty() )  ++count;
	    ++lev;
	    for( const auto* git5 : infour ) {
	      ev.goTo(git5->navType());
	      putOne((*gde), git5, ev, lev);
	      std::vector<const GeometricTimingDet*> infive= git5->components();
	      if ( infive.empty() )  ++count;
	      ++lev;
	      for( const auto* git6 : infive ) {
		ev.goTo(git6->navType());
		putOne((*gde), git6, ev, lev);
		std::vector<const GeometricTimingDet*> insix= git6->components();
		if ( insix.empty() ){
		  ++count;
		} else {
		  edm::LogError("GeometricTimingDetExtra") << "Hierarchy has exceeded hard-coded level 6 for Tracker " ;
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
    edm::ESHandle<PGeometricTimingDetExtra> pgde;
    iRecord.getRecord<PGeometricTimingDetExtraRcd>().get(pgde);
    std::map<uint32_t, const GeometricTimingDet*> helperMap;
    const GeometricTimingDet* tracker = &(*gd);
    helperMap[gd->geographicalID()] = tracker;
    std::vector<const GeometricTimingDet*> tc = tracker->components();
    for( const auto* git : tc ) { // level 1
      helperMap[git->geographicalID()] = git;
      std::vector<const GeometricTimingDet*> inone = git->components();
      for( const auto* git2 : inone ) { // level 2
	helperMap[git2->geographicalID()] = git2;
	std::vector<const GeometricTimingDet*> intwo= git2->components();
	for( const auto* git3 : intwo ) { // level 3
	  helperMap[git3->geographicalID()] = git3;
	  std::vector<const GeometricTimingDet*> inthree= git3->components();
	  for( const auto* git4 : inthree ) { // level 4
	    helperMap[git4->geographicalID()] = git4;
	    std::vector<const GeometricTimingDet*> infour= git4->components();
	    for( const auto* git5 : infour ) { // level 5
	      helperMap[git5->geographicalID()] = git5;
	      std::vector<const GeometricTimingDet*> infive= git5->components();
	      for( const auto* git6 : infive ) { // level 6
		helperMap[git6->geographicalID()] = git6;
		if ( !git6->components().empty() ){
		  edm::LogError("GeometricTimingDetExtra") << "Hierarchy has exceeded hard-coded level of 6 for Tracker " ;
		}
	      } // level 6
	    } // level 5
	  } // level 4
	} //level 3
      } // level 2
    }
  
    const std::vector<PGeometricTimingDetExtra::Item>& pgdes = pgde->pgdes_;
    gde->reserve(pgdes.size());
    std::vector<DDExpandedNode> evs; //EMPTY
    std::string nm; //EMPTY
    for (const auto & pgde : pgdes) {
      gde->emplace_back( GeometricTimingDetExtra(helperMap[pgde.geographicalId_], pgde.geographicalId_, evs
				       , pgde.volume_, pgde.density_, pgde.weight_, pgde.copy_
				       , pgde.material_, nm));
    }
  }
  return gde;
}

void MTDGeometricTimingDetExtraESModule::putOne(std::vector<GeometricTimingDetExtra> & gde, const GeometricTimingDet* gd, const DDExpandedView& ev, int lev ) {
  std::string matname = ((ev.logicalPart()).material()).name().fullname();
  std::string lpname = ((ev.logicalPart()).name().fullname());
  std::vector<DDExpandedNode> evs = GeometricTimingDetExtra::GeoHistory(ev.geoHistory().begin(),ev.geoHistory().end());
  gde.emplace_back(GeometricTimingDetExtra( gd, gd->geographicalId(), evs,
				   ((ev.logicalPart()).solid()).volume(), ((ev.logicalPart()).material()).density(),
				   ((ev.logicalPart()).material()).density() * ( ((ev.logicalPart()).solid()).volume() / 1000.),                                                                       
				   ev.copyno(), matname, lpname, true ));
}

DEFINE_FWK_EVENTSETUP_MODULE(MTDGeometricTimingDetExtraESModule);
