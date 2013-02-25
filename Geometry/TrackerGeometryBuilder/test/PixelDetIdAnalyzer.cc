#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PixelDetIdAnalyzer : public edm::EDAnalyzer {
public:
    explicit PixelDetIdAnalyzer( const edm::ParameterSet& );
    ~PixelDetIdAnalyzer() {}
  
    virtual void analyze( const edm::Event&, const edm::EventSetup& );
    
private:
};

PixelDetIdAnalyzer::PixelDetIdAnalyzer( const edm::ParameterSet& )
{}

void
PixelDetIdAnalyzer::analyze( const edm::Event& /*event*/, const edm::EventSetup& eventSetup )
{
    edm::ESHandle<TrackerGeometry> geom;
    eventSetup.get<TrackerDigiGeometryRecord>().get( geom );
    const TrackerGeometry& theTracker(*geom);

    edm::ESHandle<TrackerTopology> topo;
    eventSetup.get<IdealGeometryRecord>().get(topo);

    for( TrackerGeometry::DetContainer::const_iterator it = theTracker.detsPXF().begin(),
						      end = theTracker.detsPXF().end();
	 it != end; ++it )
    {
	const GeomDet *det = *it;

	if( det )
	{     
	    DetId detid = det->geographicalId();
	    unsigned int rawid = detid.rawId();

	    unsigned int tside = topo->pxfSide( detid );
	    unsigned int tdisk = topo->pxfDisk( detid );
	    unsigned int tblade = topo->pxfBlade( detid );
	    unsigned int tpanel = topo->pxfPanel( detid );
	    unsigned int tzindex = topo->pxfModule( detid );

	    PXFDetId idid = PXFDetId( rawid );
	    unsigned int side = PXFDetId( rawid ).side();
	    unsigned int disk = PXFDetId( rawid ).disk();
	    unsigned int blade = PXFDetId( rawid ).blade();
	    unsigned int panel = PXFDetId( rawid ).panel();
	    unsigned int module = PXFDetId( rawid ).module();
	    
	    assert( tside == side );
	    assert( tdisk == disk );
	    assert( tblade == blade );
	    assert( tpanel == panel );
	    assert( tzindex == module );
      
	    PXFDetId odid = PXFDetId( side, disk, blade, panel, module );
	    assert( idid == odid );
	    std::cout << odid << " ==  ";
	    
	    PixelEndcapName pen( detid );
	    std::string penname = pen.name();
	    
	    PXFDetId pdid = pen.getDetId();
	    assert( idid == pdid );
	    std::cout << pdid << "; " << penname.c_str() << std::endl;

	    unsigned int penside= (pen.halfCylinder() > 2 ? 2 : 1);      
	    unsigned int pendisk = pen.diskName();
	    unsigned int penblade = pen.bladeName();
	    unsigned int penpanel = pen.pannelName();
	    unsigned int penplaquette = pen.plaquetteName();

	    assert( penside == side );
	    assert( pendisk == disk );
	    assert( penblade == blade );
	    assert( penpanel == panel );
	    assert( penplaquette == module );
	}
    }
}

DEFINE_FWK_MODULE(PixelDetIdAnalyzer);
