#include "EventFilter/Phase2TrackerRawToDigi/plugins/Phase2TrackerDigiToRawProducer.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/src/fed_header.h"
#include "DataFormats/FEDRawData/src/fed_trailer.h"
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

#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>


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
    tTopo_ = tTopoHandle.product();

    // retrieve tracker geometry to get list of detids for calbling 
    edm::ESHandle< TrackerGeometry > tGeomHandle;
    es.get< TrackerDigiGeometryRecord >().get( tGeomHandle );
    tGeom_ = tGeomHandle.product();

    // // Build list of detids for dummy cabling: this should go in another producer
    // int fedid = 0;
    // int channel = 0;
    // for (auto iu = tGeom_->detUnits().begin(); iu != tGeom_->detUnits().end(); ++iu) {
    //   unsigned int detId_raw = (*iu)->geographicalId().rawId();
    //   DetId detId = DetId(detId_raw);
    //   if (detId.det() == DetId::Detector::Tracker) {
    //     if ( tTopo_->isLower(detId) != 0 ) {
    //       std::cout << tTopo_->Stack(detId) << " " << fedid << " " << channel << std::endl;
    //       channel += 1;
    //       if (channel == 72) {
    //         channel = 0;
    //         fedid += 1;
    //       }
    //     }
    //   }
    // }

    // build map of upper and lower for each module
    for (auto iu = tGeom_->detUnits().begin(); iu != tGeom_->detUnits().end(); ++iu) {
      unsigned int detId_raw = (*iu)->geographicalId().rawId();
      DetId detId = DetId(detId_raw);
      if (detId.det() == DetId::Detector::Tracker) {
          if ( tTopo_->isLower(detId) != 0 ) {
              stackMap_[tTopo_->Stack(detId)].first = detId;
          }
          if ( tTopo_->isUpper(detId) != 0 ) {
              stackMap_[tTopo_->Stack(detId)].second = detId;
          }
      }
    } // end loop on detunits
  }
  
  void Phase2TrackerDigiToRawProducer::endJob()
  {
  }
  
  void Phase2TrackerDigiToRawProducer::produce( edm::Event& event, const edm::EventSetup& es)
  {
    std::auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );
    edm::Handle< edmNew::DetSetVector<Phase2TrackerCluster1D> > digis_handle;
    event.getByToken( token_, digis_handle );
    Phase2TrackerDigiToRaw raw_producer(cabling_, tGeom_, tTopo_, stackMap_, digis_handle, 1);
    raw_producer.buildFEDBuffers(buffers);
    event.put(buffers);
  }
}
