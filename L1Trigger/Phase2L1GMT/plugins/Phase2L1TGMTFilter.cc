#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/L1TMuonPhase2/interface/SAMuon.h"
#include "Node.h"

//
// class declaration
//
using namespace Phase2L1GMT;
using namespace l1t;

class Phase2L1TGMTFilter : public edm::stream::EDProducer<> {
public:
  explicit Phase2L1TGMTFilter(const edm::ParameterSet&);
  ~Phase2L1TGMTFilter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;
  edm::EDGetTokenT<std::vector<l1t::TrackerMuon> > srcMuons_;
  bool applyLowPtFilter_;
  int ptBarrelMin_;
  int ptEndcapMin_;
};

Phase2L1TGMTFilter::Phase2L1TGMTFilter(const edm::ParameterSet& iConfig)
    : srcMuons_(consumes<std::vector<l1t::TrackerMuon> >(iConfig.getParameter<edm::InputTag>("srcMuons"))),
      applyLowPtFilter_(iConfig.getParameter<bool>("applyLowPtFilter")),
      ptBarrelMin_(iConfig.getParameter<int>("ptBarrelMin")),
      ptEndcapMin_(iConfig.getParameter<int>("ptEndcapMin")) {
  produces<std::vector<l1t::TrackerMuon> >("l1tTkMuonsGmtLowPtFix").setBranchAlias("tkMuLowPtFix");
  
}

Phase2L1TGMTFilter::~Phase2L1TGMTFilter() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void Phase2L1TGMTFilter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  Handle<std::vector<l1t::TrackerMuon> > muonHandle;
  iEvent.getByToken(srcMuons_, muonHandle);

  std::vector<l1t::TrackerMuon> out;

  for (uint i = 0; i < muonHandle->size(); ++i) {
    auto mu = muonHandle->at(i);
    bool noSAMatch = true;
    if (applyLowPtFilter_) {
      if ((fabs(mu.phEta()) < 0.9 && mu.phPt() < ptBarrelMin_) ||
          (fabs(mu.phEta()) > 0.9 && mu.phPt() < ptEndcapMin_)) {
        // if quality is already set to 0 don't continue the loop.
        for (const auto& r : mu.muonRef()) {
          if (r.isNonnull()) {
            noSAMatch = false;
            break;
          }
        }
        if (noSAMatch)
          mu.setHwQual(0);
      }
    }
    out.push_back(mu);  // store all muons otherwise
  }

  // store results
  std::unique_ptr<std::vector<l1t::TrackerMuon> > out1 = std::make_unique<std::vector<l1t::TrackerMuon> >(out);
  iEvent.put(std::move(out1),"l1tTkMuonsGmtLowPtFix");
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void Phase2L1TGMTFilter::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void Phase2L1TGMTFilter::endStream() {}

void Phase2L1TGMTFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2L1TGMTFilter);
