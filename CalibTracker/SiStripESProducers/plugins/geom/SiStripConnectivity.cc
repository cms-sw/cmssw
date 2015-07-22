#include "CalibTracker/SiStripESProducers/plugins/geom/SiStripConnectivity.h"
//#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <memory>

SiStripConnectivity::SiStripConnectivity(const edm::ParameterSet& p) {
  //the following lines are needed to tell the framework what data is being produced
  setWhatProduced(this, &SiStripConnectivity::produceFecCabling);
  setWhatProduced(this, &SiStripConnectivity::produceDetCabling);
}

SiStripConnectivity::~SiStripConnectivity() {
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

// ------------ methods called to produce the data  ------------

std::auto_ptr<SiStripFecCabling> SiStripConnectivity::produceFecCabling( const SiStripFecCablingRcd& iRecord ){
  edm::ESHandle<SiStripFedCabling> pDD;
  iRecord.getRecord<SiStripFedCablingRcd>().get(pDD );
  //here build an object of type SiStripFecCabling using  **ONLY** the information from class SiStripFedCabling, 
  SiStripFecCabling * FecConnections = new SiStripFecCabling( *(pDD.product()));
  return std::auto_ptr<SiStripFecCabling>( FecConnections );
}

std::auto_ptr<SiStripDetCabling> SiStripConnectivity::produceDetCabling( const SiStripDetCablingRcd& iRecord ){
  edm::ESHandle<SiStripFedCabling> pDD;
  iRecord.getRecord<SiStripFedCablingRcd>().get(pDD );
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iRecord.getRecord<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
  //here build an object of type SiStripDetCabling using  **ONLY** the information from class SiStripFedCabling, 
  SiStripDetCabling * DetConnections = new SiStripDetCabling( *(pDD.product()),tTopo);
  return std::auto_ptr<SiStripDetCabling>( DetConnections );
}

//define this as a plug-in
// DEFINE_FWK_EVENTSETUP_MODULE(SiStripConnectivity);
