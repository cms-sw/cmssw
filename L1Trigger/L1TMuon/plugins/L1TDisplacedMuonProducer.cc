// system include files
#include <memory>
#include <fstream>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

#include "L1Trigger/L1TMuon/src/Phase2/L1TDisplacedMuonBuilder.h"
//
// class declaration
//
using namespace l1t;

class L1TDisplacedMuonProducer : public edm::EDProducer
{
public:
  explicit L1TDisplacedMuonProducer(const edm::ParameterSet&);
  ~L1TDisplacedMuonProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

  // ----------member data ---------------------------
  edm::InputTag cscCompTag_;
  edm::InputTag cscLctTag_;
  edm::InputTag padTag_;
  edm::InputTag copadTag_;
  edm::InputTag me0TriggerTag_;
  edm::InputTag me0SegmentTag_;
  edm::InputTag emtfTag_;
  edm::InputTag muonTag_;
  edm::InputTag bmtfTag_;
  edm::InputTag omtfNegTag_;
  edm::InputTag omtfPosTag_;
  edm::InputTag emtfNegTag_;
  edm::InputTag emtfPosTag_;

  edm::EDGetTokenT<CSCComparatorDigiCollection> comparatorToken_;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> lctToken_;
  edm::EDGetTokenT<GEMPadDigiCollection> padToken_;
  edm::EDGetTokenT<GEMCoPadDigiCollection> copadToken_;
  edm::EDGetTokenT<ME0TriggerDigiCollection> me0triggerToken_;
  edm::EDGetTokenT<ME0SegmentCollection> segmentToken_;
  edm::EDGetTokenT<l1t::EMTFTrackCollection> emtfToken_;
  edm::EDGetTokenT<l1t::MuonBxCollection> muonToken_;
  edm::EDGetTokenT<l1t::MuonBxCollection> bmtfToken_;
  edm::EDGetTokenT<l1t::MuonBxCollection> emtfNegToken_;
  edm::EDGetTokenT<l1t::MuonBxCollection> emtfPosToken_;
  edm::EDGetTokenT<l1t::MuonBxCollection> omtfNegToken_;
  edm::EDGetTokenT<l1t::MuonBxCollection> omtfPosToken_;

  edm::ParameterSet config_;
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
L1TDisplacedMuonProducer::L1TDisplacedMuonProducer(const edm::ParameterSet& iConfig)
{
  cscCompTag_ = iConfig.getParameter<edm::InputTag>("cscCompTag");
  cscLctTag_ = iConfig.getParameter<edm::InputTag>("cscLctTag");
  padTag_ = iConfig.getParameter<edm::InputTag>("padTag");
  copadTag_ = iConfig.getParameter<edm::InputTag>("copadTag");
  me0TriggerTag_ = iConfig.getParameter<edm::InputTag>("me0TriggerTag");
  me0SegmentTag_ = iConfig.getParameter<edm::InputTag>("me0SegmentTag");
  emtfTag_ = iConfig.getParameter<edm::InputTag>("emtfTag");
  muonTag_ = iConfig.getParameter<edm::InputTag>("muonTag");
  bmtfTag_ = iConfig.getParameter<edm::InputTag>("bmtfTag");
  omtfNegTag_ = iConfig.getParameter<edm::InputTag>("omtfNegTag");
  omtfPosTag_ = iConfig.getParameter<edm::InputTag>("omtfPosTag");
  emtfNegTag_ = iConfig.getParameter<edm::InputTag>("emtfNegTag");
  emtfPosTag_ = iConfig.getParameter<edm::InputTag>("emtfPosTag");

  comparatorToken_ = consumes<CSCComparatorDigiCollection>(cscCompTag_);
  lctToken_ = consumes<CSCCorrelatedLCTDigiCollection>(cscLctTag_);
  padToken_ = consumes<GEMPadDigiCollection>(padTag_);
  copadToken_ = consumes<GEMCoPadDigiCollection>(copadTag_);
  me0triggerToken_ = consumes<ME0TriggerDigiCollection>(me0TriggerTag_);
  segmentToken_ = consumes<ME0SegmentCollection>(me0SegmentTag_);
  emtfToken_ = consumes<l1t::EMTFTrackCollection>(emtfTag_);
  muonToken_ = consumes<l1t::MuonBxCollection>(muonTag_);
  bmtfToken_ = consumes<l1t::MuonBxCollection>(bmtfTag_);
  omtfNegToken_ = consumes<l1t::MuonBxCollection>(omtfNegTag_);
  omtfPosToken_ = consumes<l1t::MuonBxCollection>(omtfPosTag_);
  emtfNegToken_ = consumes<l1t::MuonBxCollection>(emtfNegTag_);
  emtfPosToken_ = consumes<l1t::MuonBxCollection>(emtfPosTag_);

  config_ = iConfig;

  //register your products
  produces<l1t::MuonBxCollection>("NoVtx");
}

L1TDisplacedMuonProducer::~L1TDisplacedMuonProducer()
{
}


//
// member functions
//



// ------------ method called to produce the data  ------------
void
L1TDisplacedMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // construct a new builder
  std::unique_ptr<L1TMuon::L1TDisplacedMuonBuilder>
    builder( new L1TMuon::L1TDisplacedMuonBuilder(config_) );

  // get the trigger geometry
  edm::ESHandle<CSCGeometry> h;
  iSetup.get<MuonGeometryRecord>().get(h);
  builder->setCSCGeometry(&*h);

  edm::ESHandle<GEMGeometry> h_gem;
  iSetup.get<MuonGeometryRecord>().get(h_gem);
  builder->setGEMGeometry(&*h_gem);

  edm::ESHandle<RPCGeometry> h_rpc;
  iSetup.get<MuonGeometryRecord>().get(h_rpc);
  builder->setRPCGeometry(&*h_rpc);

  edm::ESHandle<DTGeometry> h_dt;
  iSetup.get<MuonGeometryRecord>().get(h_dt);
  builder->setDTGeometry(&*h_dt);

  edm::ESHandle<ME0Geometry> h_me0;
  iSetup.get<MuonGeometryRecord>().get(h_me0);
  builder->setME0Geometry(&*h_me0);

  // input collections
  edm::Handle<CSCComparatorDigiCollection> comparators;
  iEvent.getByToken(comparatorToken_, comparators);

  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts;
  iEvent.getByToken(lctToken_, lcts);

  edm::Handle<GEMPadDigiCollection> pads;
  iEvent.getByToken(padToken_, pads);

  edm::Handle<GEMCoPadDigiCollection> copads;
  iEvent.getByToken(copadToken_, copads);

  edm::Handle<ME0SegmentCollection> segments;
  iEvent.getByToken(segmentToken_, segments);

  edm::Handle<l1t::MuonBxCollection> inMuonsPhase2;
  iEvent.getByToken(muonToken_, inMuonsPhase2);

  edm::Handle<l1t::EMTFTrackCollection> emtfTracks;
  iEvent.getByToken(emtfToken_, emtfTracks);

  edm::Handle<l1t::MuonBxCollection> inBmtf;
  iEvent.getByToken(bmtfToken_, inBmtf);

  edm::Handle<l1t::MuonBxCollection> inOmtfPos;
  iEvent.getByToken(omtfPosToken_, inOmtfPos);

  edm::Handle<l1t::MuonBxCollection> inOmtfNeg;
  iEvent.getByToken(omtfNegToken_, inOmtfNeg);

  edm::Handle<l1t::MuonBxCollection> inEmtfNeg;
  iEvent.getByToken(emtfNegToken_, inEmtfNeg);

  edm::Handle<l1t::MuonBxCollection> inEmtfPos;
  iEvent.getByToken(emtfPosToken_, inEmtfPos);

  // new output collection
  std::unique_ptr<l1t::MuonBxCollection> outMuonsPhase2 (new l1t::MuonBxCollection());

  // build the displaced muons
  builder->build(comparators.product(),
                 lcts.product(),
                 pads.product(),
                 copads.product(),
                 segments.product(),
                 emtfTracks.product(),
                 inMuonsPhase2,
                 inBmtf, inOmtfPos, inOmtfNeg, inEmtfNeg, inEmtfPos,
                 outMuonsPhase2);

  // put output collection in event
  iEvent.put(std::move(outMuonsPhase2),"NoVtx");
}

// ------------ method called once each job just before starting event loop  ------------
void
L1TDisplacedMuonProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TDisplacedMuonProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
L1TDisplacedMuonProducer::beginRun(edm::Run const& run, edm::EventSetup const& iSetup)
{
}

// ------------ method called when ending the processing of a run  ------------
void
L1TDisplacedMuonProducer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void
L1TDisplacedMuonProducer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void
L1TDisplacedMuonProducer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TDisplacedMuonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TDisplacedMuonProducer);
