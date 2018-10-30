#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"


#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanAlgo.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanTrackFinder.h"
#include "DataFormats/L1TMuon/interface/L1MuKBMTCombinedStub.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

//
// class declaration
//

class L1TMuonBarrelKalmanTrackProducer : public edm::stream::EDProducer<> {
   public:
      explicit L1TMuonBarrelKalmanTrackProducer(const edm::ParameterSet&);
      ~L1TMuonBarrelKalmanTrackProducer() override;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      void beginStream(edm::StreamID) override;
      void produce(edm::Event&, const edm::EventSetup&) override;
      void endStream() override;
  edm::EDGetTokenT<std::vector<L1MuKBMTCombinedStub> > src_;
  std::vector<int> bx_;
  L1TMuonBarrelKalmanAlgo *algo_;
  L1TMuonBarrelKalmanTrackFinder *trackFinder_;

  
  
  

};
L1TMuonBarrelKalmanTrackProducer::L1TMuonBarrelKalmanTrackProducer(const edm::ParameterSet& iConfig):
  src_(consumes<std::vector<L1MuKBMTCombinedStub> >(iConfig.getParameter<edm::InputTag>("src"))),
  bx_(iConfig.getParameter<std::vector<int> >("bx")),
  algo_(new L1TMuonBarrelKalmanAlgo(iConfig.getParameter<edm::ParameterSet>("algoSettings"))),
  trackFinder_(new L1TMuonBarrelKalmanTrackFinder(iConfig.getParameter<edm::ParameterSet>("trackFinderSettings")))
{
  produces <L1MuKBMTrackBxCollection>();
  produces <l1t::RegionalMuonCandBxCollection>("BMTF");

}


L1TMuonBarrelKalmanTrackProducer::~L1TMuonBarrelKalmanTrackProducer()
{
 
  if (algo_!=nullptr)
    delete algo_;

  if (trackFinder_!=nullptr)
    delete trackFinder_;
    
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}

 


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1TMuonBarrelKalmanTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   Handle<std::vector<L1MuKBMTCombinedStub> >stubHandle;
   iEvent.getByToken(src_,stubHandle);

   L1MuKBMTCombinedStubRefVector stubs;
   for (uint i=0;i<stubHandle->size();++i) {
     L1MuKBMTCombinedStubRef r(stubHandle,i);
     stubs.push_back(r);
   }


   std::unique_ptr<l1t::RegionalMuonCandBxCollection> outBMTF(new l1t::RegionalMuonCandBxCollection());
   std::unique_ptr<L1MuKBMTrackBxCollection> out(new L1MuKBMTrackBxCollection());
   outBMTF->setBXRange(bx_.front(),bx_.back());
   out->setBXRange(bx_.front(),bx_.back());
   for (const auto& bx : bx_) {
     L1MuKBMTrackCollection tmp = trackFinder_->process(algo_,stubs,bx);
     for (const auto& track :tmp) {
       out->push_back(bx,track);
       algo_->addBMTFMuon(bx,track,outBMTF);
     } 
   }
   iEvent.put(std::move(outBMTF),"BMTF");
   iEvent.put(std::move(out));

}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
L1TMuonBarrelKalmanTrackProducer::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
L1TMuonBarrelKalmanTrackProducer::endStream() {
}

void
L1TMuonBarrelKalmanTrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonBarrelKalmanTrackProducer);
