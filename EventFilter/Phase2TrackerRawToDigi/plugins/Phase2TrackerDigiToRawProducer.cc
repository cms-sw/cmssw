#include "EventFilter/Phase2TrackerRawToDigi/plugins/Phase2TrackerDigiToRawProducer.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/src/fed_header.h"
#include "DataFormats/FEDRawData/src/fed_trailer.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
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

#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

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
    // get tracker topology
    edm::ESHandle<TrackerTopology> tTopoHand;
    es.get<IdealGeometryRecord>().get(tTopoHand);
    topo_ = tTopoHand.product();
  }
  
  void Phase2TrackerDigiToRawProducer::endJob()
  {
  }
  
  void Phase2TrackerDigiToRawProducer::produce( edm::Event& event, const edm::EventSetup& es)
  {
    std::auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );
    edm::Handle< edmNew::DetSetVector<SiPixelCluster> > digis_handle;
    event.getByLabel("siPixelClusters","", digis_handle);
    // temp : get list of detids for modules in tracker
    const edmNew::DetSetVector<SiPixelCluster>* digs = digis_handle.product();
    for (edmNew::DetSetVector<SiPixelCluster>::const_iterator it = digs->begin(); it != digs->end(); it++)
    {
      SiStripDetId did(it->detId());
      std::cout << it->detId() << " " << did.subDetector() << std::endl;
    }
    // end of temp
    Phase2TrackerDigiToRaw raw_producer(cabling_, topo_, digis_handle, 1);
    raw_producer.buildFEDBuffers(buffers);
    event.put(buffers);
  }
}
