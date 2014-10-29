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

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

// to use stacked geometry
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h" 


using namespace std;

namespace Phase2Tracker {

  Phase2TrackerDigiToRawProducer::Phase2TrackerDigiToRawProducer( const edm::ParameterSet& pset ) :
    cabling_(0)
  {
    token_ = consumes<edmNew::DetSetVector<SiPixelCluster>>(pset.getParameter<edm::InputTag>("ProductLabel"));
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
    // build map to associate stacked tracker detids
    edm::ESHandle<StackedTrackerGeometry>  StackedGeometryHandle;
    es.get<StackedTrackerGeometryRecord>().get(StackedGeometryHandle);
    const StackedTrackerGeometry * stackedGeometry = StackedGeometryHandle.product();
    int id1, id2;
    for (auto stk = stackedGeometry->stacks().begin(); stk != stackedGeometry->stacks().end(); ++stk)
    {
      StackedTrackerDetUnit* stackDetUnit = (*stk);
      id1 = stackDetUnit->stackMember(0);
      id2 = stackDetUnit->stackMember(1);
      stackMap_[id1] =  id2;
      stackMap_[id2] = -id1;
    }

  }
  
  void Phase2TrackerDigiToRawProducer::endJob()
  {
  }
  
  void Phase2TrackerDigiToRawProducer::produce( edm::Event& event, const edm::EventSetup& es)
  {
    std::auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );
    edm::Handle< edmNew::DetSetVector<SiPixelCluster> > digis_handle;
    event.getByToken( token_, digis_handle );
    Phase2TrackerDigiToRaw raw_producer(cabling_, stackMap_, digis_handle, 1);
    raw_producer.buildFEDBuffers(buffers);
    event.put(buffers);
  }
}
