#include "EventFilter/Phase2TrackerRawToDigi/plugins/Phase2TrackerDigiToRawProducer.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/src/fed_header.h"
#include "DataFormats/FEDRawData/src/fed_trailer.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerDigiToRaw.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/utils.h"
#include "CondFormats/DataRecord/interface/Phase2TrackerCablingRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ext/algorithm>

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/DetId/interface/DetId.h"

// to use geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"


using namespace std;

namespace Phase2Tracker {

  Phase2TrackerDigiToRawProducer::Phase2TrackerDigiToRawProducer( const edm::ParameterSet& pset ) :
    cabling_(0)
  {
    token_ = consumes<edmNew::DetSetVector<Phase2TrackerCluster1D>>(pset.getParameter<edm::InputTag>("ProductLabel"));
    produces<FEDRawDataCollection>();
  }
  
  Phase2TrackerDigiToRawProducer::~Phase2TrackerDigiToRawProducer()
  {
  }
  
  void Phase2TrackerDigiToRawProducer::beginJob( )
  {
  }
  
  void Phase2TrackerDigiToRawProducer::beginRun( edm::Run const& run, edm::EventSetup const& es)
  {
    // fetch cabling from event setup
    edm::ESHandle<Phase2TrackerCabling> c;
    es.get<Phase2TrackerCablingRcd>().get(c);
    cabling_ = c.product();

    // retrieve tracker topology 
    edm::ESHandle<TrackerTopology> tTopoHandle;
    es.get<IdealGeometryRecord>().get(tTopoHandle);
    const TrackerTopology* const tTopo = tTopoHandle.product();

    // retrieve tracker geometry 
    edm::ESHandle< TrackerGeometry > tGeomHandle;
    es.get< TrackerDigiGeometryRecord >().get( tGeomHandle );
    const TrackerGeometry* const theTrackerGeom = tGeomHandle.product();

    // build map to associate stacked tracker detids 
    for (TrackerGeometry::DetIdContainer::const_iterator gd = theTrackerGeom->detIds().begin(); gd != theTrackerGeom->detIds().end(); gd++) {
        DetId id = gd->rawId();
        // // get detids and layers to build cabling file
        // if (id.subdetId() == StripSubdetector::TOB and tTopo->PartnerDetId(id) == 0) {
        //     std::cout << int(id) << " " << tTopo->layer(id) << std::endl;
        // } else if (id.subdetId() == StripSubdetector::TID) {
        //     std::cout << int(id) << " on disk  " << tTopo->side(id) << std::endl;
        // }
        if (tTopo->PartnerDetId(id) != 0) {
            if(tTopo->isLower(id)) {
                stackMap_[id] = tTopo->PartnerDetId(id);
            } else {
                stackMap_[id] = -tTopo->PartnerDetId(id);
            }
        }
    }
  }
  
  void Phase2TrackerDigiToRawProducer::endJob()
  {
  }
  
  void Phase2TrackerDigiToRawProducer::produce( edm::Event& event, const edm::EventSetup& es)
  {
    std::auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );
    edm::Handle< edmNew::DetSetVector<Phase2TrackerCluster1D> > digis_handle;
    event.getByToken( token_, digis_handle );
    Phase2TrackerDigiToRaw raw_producer(cabling_, stackMap_, digis_handle, 1);
    raw_producer.buildFEDBuffers(buffers);
    event.put(buffers);
  }
}
