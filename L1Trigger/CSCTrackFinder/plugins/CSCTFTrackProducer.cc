#include "L1Trigger/CSCTrackFinder/plugins/CSCTFTrackProducer.h"

#include <vector>
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"

CSCTFTrackProducer::CSCTFTrackProducer(const edm::ParameterSet& pset)
    : my_dtrc{},
      useDT{pset.getParameter<bool>("useDT")},
      TMB07{pset.getParameter<bool>("isTMB07")},
      readDtDirect{pset.getParameter<bool>("readDtDirect")},
      input_module{
          consumes<CSCCorrelatedLCTDigiCollection>(pset.getUntrackedParameter<edm::InputTag>("SectorReceiverInput"))},
      dt_producer{consumes<L1MuDTChambPhContainer>(pset.getUntrackedParameter<edm::InputTag>("DTproducer"))},
      directProd{
          consumes<CSCTriggerContainer<csctf::TrackStub> >(pset.getUntrackedParameter<edm::InputTag>("DtDirectProd"))},
      m_scalesToken(esConsumes<L1MuTriggerScales, L1MuTriggerScalesRcd>()),
      m_ptScaleToken(esConsumes<L1MuTriggerPtScale, L1MuTriggerPtScaleRcd>()),
      m_pDDToken(esConsumes<CSCGeometry, MuonGeometryRecord>()),
      sp_pset{pset.getParameter<edm::ParameterSet>("SectorProcessor")},
      m_builderTokens(CSCTFTrackBuilder::consumes(sp_pset, consumesCollector())) {
  m_scalesCacheID = 0ULL;
  m_ptScaleCacheID = 0ULL;
  produces<L1CSCTrackCollection>();
  produces<CSCTriggerContainer<csctf::TrackStub> >();

  //CSCSectorReceiverLUT has a static member it fills by reading a file
  usesResource("CSCSectorReceiverLUT");
  //CSCTFSPcoreLogic has non-const static members
  usesResource("CSCTFSPCoreLogic");
  //CSCTFPtLUT has a static member it fills by reading a file
  usesResource("CSCTFPtLUT");
}

void CSCTFTrackProducer::produce(edm::Event& e, const edm::EventSetup& c) {
  // Update CSCTFTrackBuilder only if the scales have changed.  Use the
  // EventSetup cacheIdentifier to tell when this has happened.
  if (c.get<L1MuTriggerScalesRcd>().cacheIdentifier() != m_scalesCacheID ||
      c.get<L1MuTriggerPtScaleRcd>().cacheIdentifier() != m_ptScaleCacheID) {
    edm::ESHandle<L1MuTriggerScales> scales = c.getHandle(m_scalesToken);

    edm::ESHandle<L1MuTriggerPtScale> ptScale = c.getHandle(m_ptScaleToken);

    my_builder = std::make_unique<CSCTFTrackBuilder>(sp_pset, TMB07, scales.product(), ptScale.product());
    my_builder->initialize(c, m_builderTokens);

    m_scalesCacheID = c.get<L1MuTriggerScalesRcd>().cacheIdentifier();
    m_ptScaleCacheID = c.get<L1MuTriggerPtScaleRcd>().cacheIdentifier();
  }

  // set geometry pointer
  edm::ESHandle<CSCGeometry> pDD = c.getHandle(m_pDDToken);

  edm::Handle<CSCCorrelatedLCTDigiCollection> LCTs;
  std::unique_ptr<L1CSCTrackCollection> track_product(new L1CSCTrackCollection);
  e.getByToken(input_module, LCTs);
  std::unique_ptr<CSCTriggerContainer<csctf::TrackStub> > dt_stubs(new CSCTriggerContainer<csctf::TrackStub>);

  // Either emulate or directly read in DT stubs based on switch
  //////////////////////////////////////////////////////////////
  CSCTriggerContainer<csctf::TrackStub> emulStub;
  if (readDtDirect == false) {
    edm::Handle<L1MuDTChambPhContainer> dttrig;
    e.getByToken(dt_producer, dttrig);
    emulStub = my_dtrc.process(dttrig.product());
  } else {
    edm::Handle<CSCTriggerContainer<csctf::TrackStub> > stubsFromDaq;
    //e.getByLabel("csctfunpacker","DT",stubsFromDaq);
    e.getByToken(directProd, stubsFromDaq);
    const CSCTriggerContainer<csctf::TrackStub>* stubPointer = stubsFromDaq.product();
    emulStub.push_many(*stubPointer);
  }

  my_builder->buildTracks(LCTs.product(), (useDT ? &emulStub : nullptr), track_product.get(), dt_stubs.get());

  e.put(std::move(track_product));
  e.put(std::move(dt_stubs));
}
