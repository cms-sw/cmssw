/** \class ReadRecHit
 *
 * ReadRecHit is a analyzer which reads rechits
 *
 * \author C. Genta
 *
 */

// system includes
#include <memory>
#include <string>
#include <iostream>

// user includes
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiStripRecHitConverter/test/ReadRecHitAlgorithm.h"

namespace cms {
  class ReadRecHit : public edm::one::EDAnalyzer<> {
  public:
    explicit ReadRecHit(const edm::ParameterSet& conf);
    virtual ~ReadRecHit() override = default;

    virtual void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
    ReadRecHitAlgorithm readRecHitAlgorithm_;
    const std::string recHitProducer_;
    const edm::EDGetTokenT<SiStripMatchedRecHit2DCollection> matchedRecHitToken_;
    const edm::EDGetTokenT<SiStripRecHit2DCollection> rphiToken_;
    const edm::EDGetTokenT<SiStripRecHit2DCollection> stereoToken_;
  };
}  // namespace cms

namespace cms {

  ReadRecHit::ReadRecHit(edm::ParameterSet const& conf)
      : readRecHitAlgorithm_(conf),
        recHitProducer_(conf.getParameter<std::string>("RecHitProducer")),
        matchedRecHitToken_(
            consumes<SiStripMatchedRecHit2DCollection>(edm::InputTag{recHitProducer_, "matchedRecHit"})),
        rphiToken_(consumes<SiStripRecHit2DCollection>(edm::InputTag{recHitProducer_, "rphiRecHit"})),
        stereoToken_(consumes<SiStripRecHit2DCollection>(edm::InputTag{recHitProducer_, "stereoRecHit"})) {}

  // Functions that gets called by framework every event
  void ReadRecHit::analyze(const edm::Event& e, const edm::EventSetup& es) {
    using namespace edm;

    // Step A: Get Inputs
    edm::Handle<SiStripMatchedRecHit2DCollection> rechitsmatched = e.getHandle(matchedRecHitToken_);
    edm::Handle<SiStripRecHit2DCollection> rechitsrphi = e.getHandle(rphiToken_);
    edm::Handle<SiStripRecHit2DCollection> rechitsstereo = e.getHandle(stereoToken_);

    edm::LogInfo("ReadRecHit") << "Matched hits:";
    readRecHitAlgorithm_.run(rechitsmatched.product());
    edm::LogInfo("ReadRecHit") << "Rphi hits:";
    readRecHitAlgorithm_.run(rechitsrphi.product());
    edm::LogInfo("ReadRecHit") << "Stereo hits:";
    readRecHitAlgorithm_.run(rechitsstereo.product());
  }

}  // namespace cms
