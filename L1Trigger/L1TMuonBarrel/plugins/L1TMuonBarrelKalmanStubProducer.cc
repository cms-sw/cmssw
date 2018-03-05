#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"


#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanStubProcessor.h"



//
// class declaration
//

class L1TMuonBarrelKalmanStubProducer : public edm::stream::EDProducer<> {
   public:
      explicit L1TMuonBarrelKalmanStubProducer(const edm::ParameterSet&);
      ~L1TMuonBarrelKalmanStubProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;
  edm::EDGetTokenT<L1MuDTChambPhContainer> srcPhi_;
  edm::EDGetTokenT<L1MuDTChambThContainer> srcTheta_;
  L1TMuonBarrelKalmanStubProcessor * proc_;

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
L1TMuonBarrelKalmanStubProducer::L1TMuonBarrelKalmanStubProducer(const edm::ParameterSet& iConfig):
  srcPhi_(consumes<L1MuDTChambPhContainer>(iConfig.getParameter<edm::InputTag>("srcPhi"))),
  srcTheta_(consumes<L1MuDTChambThContainer>(iConfig.getParameter<edm::InputTag>("srcTheta"))),
  proc_(new L1TMuonBarrelKalmanStubProcessor(iConfig))
{
  produces <L1MuKBMTCombinedStubCollection>();
}


L1TMuonBarrelKalmanStubProducer::~L1TMuonBarrelKalmanStubProducer()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)
  if (proc_!=0)
    delete proc_;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1TMuonBarrelKalmanStubProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   Handle<L1MuDTChambPhContainer> phiIn;
   iEvent.getByToken(srcPhi_,phiIn);

   Handle<L1MuDTChambThContainer> thetaIn;
   iEvent.getByToken(srcTheta_,thetaIn);

   L1MuKBMTCombinedStubCollection stubs = proc_->makeStubs(phiIn.product(),thetaIn.product());
   iEvent.put(std::make_unique<L1MuKBMTCombinedStubCollection>(stubs));
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
L1TMuonBarrelKalmanStubProducer::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
L1TMuonBarrelKalmanStubProducer::endStream() {
}

void
L1TMuonBarrelKalmanStubProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonBarrelKalmanStubProducer);
