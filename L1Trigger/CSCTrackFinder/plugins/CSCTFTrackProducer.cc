#include "CSCTFTrackProducer.h"

#include "L1Trigger/CSCTrackFinder/src/CSCTFTrackBuilder.h"

#include <vector>
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"

#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"


CSCTFTrackProducer::CSCTFTrackProducer(const edm::ParameterSet& pset)
{
  input_module = pset.getUntrackedParameter<edm::InputTag>("SectorReceiverInput");
  edm::ParameterSet sp_pset = pset.getParameter<edm::ParameterSet>("SectorProcessor");
  my_builder = new CSCTFTrackBuilder(sp_pset);
  produces<L1CSCTrackCollection>();
  produces<CSCTriggerContainer<csctf::TrackStub> >();
}

CSCTFTrackProducer::~CSCTFTrackProducer()
{
  delete my_builder;
  my_builder = NULL;
}

void CSCTFTrackProducer::produce(edm::Event & e, const edm::EventSetup& c)
{
  // set geometry pointer
  edm::ESHandle<CSCGeometry> pDD;

  c.get<MuonGeometryRecord>().get( pDD );
  CSCTriggerGeometry::setGeometry(pDD);

  edm::Handle<CSCCorrelatedLCTDigiCollection> LCTs;
  edm::Handle<L1MuDTChambPhContainer> dttrig;
  std::auto_ptr<L1CSCTrackCollection> track_product(new L1CSCTrackCollection);
  std::auto_ptr<CSCTriggerContainer<csctf::TrackStub> > dt_stubs(new CSCTriggerContainer<csctf::TrackStub>);

  e.getByLabel(input_module.label(),input_module.instance(), LCTs);
  e.getByType(dttrig);

  my_builder->buildTracks(LCTs.product(), dttrig.product(), track_product.get(), dt_stubs.get());

  e.put(track_product);
  e.put(dt_stubs);
}
