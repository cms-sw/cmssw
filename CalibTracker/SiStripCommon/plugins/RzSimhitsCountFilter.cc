#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "boost/foreach.hpp"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class RzSimhitCountFilter : public edm::EDFilter 
{
 public:
  explicit RzSimhitCountFilter(const edm::ParameterSet& cfg) : 
    inputTags_(cfg.getParameter<std::vector<edm::InputTag> >("InputTags")),
    max_radius_(cfg.getParameter<double>("MaxRadius")),
    max_z_(cfg.getParameter<double>("MaxZ")),
    min_hits_(cfg.getParameter<uint32_t>("MinHits"))
  { }
        
 private:

  const std::vector<edm::InputTag> inputTags_;
  const double max_radius_, max_z_;
  const unsigned min_hits_;


  bool filter(edm::Event& evt, const edm::EventSetup& es) {
    edm::ESHandle<TrackerGeometry> theTrackerGeometry;         
    es.get<TrackerDigiGeometryRecord>().get( theTrackerGeometry );  

    unsigned count = 0;
    BOOST_FOREACH(edm::InputTag input, inputTags_) { edm::Handle<std::vector<PSimHit> > simhits; evt.getByLabel(input, simhits);
      BOOST_FOREACH( const PSimHit hit, *simhits ) {
	const StripGeomDetUnit* sgdu = dynamic_cast<const StripGeomDetUnit*>( theTrackerGeometry->idToDet( hit.detUnitId() ) );
	const GlobalPoint position = sgdu->toGlobal(hit.localPosition());

	if( position.transverse() < max_radius_ && fabs(position.z()) < max_z_ ) count++;
	if( count >= min_hits_ ) return true;
      }
    }
    return false;
  }    
};

DEFINE_FWK_MODULE( RzSimhitCountFilter );
