// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

//
// class declaration
//
namespace {
  const float dummy = -9.;
  const GlobalPoint dummyGP(dummy,dummy,0.);
} //end anonymous namespace

class VertexCompositeCandidateCollectionSelector : public edm::stream::EDProducer<> {
public:
  explicit VertexCompositeCandidateCollectionSelector(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      void produce(edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------

  edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> v0Token_;
  edm::EDGetTokenT<reco::BeamSpot>         bsToken_;
  edm::EDGetTokenT<reco::VertexCollection> pvToken_;

  int pvNDOF_;

  std::string label_;

  // list of variables for the selection
  float lxyCUT_;
  float lxyWRTbsCUT_;
  bool  debug_;
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
VertexCompositeCandidateCollectionSelector::VertexCompositeCandidateCollectionSelector(const edm::ParameterSet& iConfig)
  : v0Token_ ( consumes<reco::VertexCompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("v0") )            )
  , bsToken_ ( consumes<reco::BeamSpot>                          (iConfig.getParameter<edm::InputTag>("beamSpot") )      )
  , pvToken_ ( consumes<reco::VertexCollection>                  (iConfig.getParameter<edm::InputTag>("primaryVertex") ) )
  , pvNDOF_ ( iConfig.getParameter<int> ("pvNDOF") )
  , label_  ( iConfig.getParameter<edm::InputTag>("v0").instance() )
  , lxyCUT_      ( iConfig.getParameter<double>("lxyCUT") )
  , lxyWRTbsCUT_ ( iConfig.getParameter<double>("lxyWRTbsCUT") )
  , debug_ ( iConfig.getUntrackedParameter<bool>("debug") )
{
  if (debug_) std::cout << "VertexCompositeCandidateCollectionSelector::VertexCompositeCandidateCollectionSelector" << std::endl;
  // product
  produces<reco::VertexCompositeCandidateCollection>();

  //now do what ever other initialization is needed
  
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
VertexCompositeCandidateCollectionSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
  if (debug_) std::cout << "VertexCompositeCandidateCollectionSelector::produce" << std::endl;

   // Create auto_ptr for each collection to be stored in the Event
   auto result = std::make_unique<reco::VertexCompositeCandidateCollection>();

   edm::Handle<reco::BeamSpot> beamspotHandle;
   iEvent.getByToken(bsToken_,beamspotHandle);
   reco::BeamSpot const * bs = nullptr;
   if (beamspotHandle.isValid())
     bs = &(*beamspotHandle);


   edm::Handle< reco::VertexCollection > pvHandle;
   iEvent.getByToken(pvToken_, pvHandle );
   reco::Vertex const * pv = nullptr;  
   if (pvHandle.isValid()) {
     pv = &pvHandle->front();
     //--- pv fake (the pv collection should have size==1 and the pv==beam spot)
     if (   pv->isFake() || pv->tracksSize()==0
	    // definition of goodOfflinePrimaryVertex
	    || pv->ndof() < pvNDOF_ || pv->z() > 24.)  pv = nullptr;
   }

   edm::Handle<reco::VertexCompositeCandidateCollection> v0Handle;
   iEvent.getByToken(v0Token_, v0Handle);
   int n = ( v0Handle.isValid() ? v0Handle->size() : -1 );
   if (debug_) std::cout << "n: " << n << std::endl;
   if (n>0) {
  
     auto const& v0s = *v0Handle.product();    
     for ( auto const& v0 : v0s ) {
       GlobalPoint displacementFromPV2D = ( pv==nullptr ? dummyGP : GlobalPoint( (pv->x() - v0.vx()), 
										 (pv->y() - v0.vy()), 
										 0. ) );
       GlobalPoint displacementFromBS2D = ( bs==nullptr ? dummyGP : GlobalPoint( v0.vx() - bs->x(v0.vz()),
										 v0.vy() - bs->y(v0.vz()),
										 0. ) );
       float abslxy      = ( pv==nullptr ? dummy : displacementFromPV2D.perp() );
       float abslxyWRTbs = ( bs==nullptr ? dummy : displacementFromBS2D.perp() );

       if (debug_) std::cout << "abslxy: " << abslxy << " w.r.t. " << lxyCUT_ << " ==> " << ( abslxy >= lxyCUT_ ? "OK" : "KO" ) << std::endl;
       if (debug_) std::cout << "abslxyWRTbs: " << abslxyWRTbs << " w.r.t. " << lxyWRTbsCUT_ << " ==> " << ( abslxyWRTbs >= lxyWRTbsCUT_ ? "OK" : "KO" ) << std::endl;       
       if (abslxy      < lxyCUT_     ) continue;
       if (abslxyWRTbs < lxyWRTbsCUT_) continue;
       result->push_back(v0);
     }
   }

   if (debug_) std::cout << "result: " << result->size() << std::endl;
   // put into the Event
   // Write the collections to the Event
   result->shrink_to_fit(); iEvent.put(std::move(result) );
 
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
VertexCompositeCandidateCollectionSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("v0");
  desc.add<edm::InputTag>("beamSpot");
  desc.add<edm::InputTag>("primaryVertex");
  desc.add<int>("pvNDOF");
  desc.add<double>("lxyCUT",      16.); // cm (2016 pixel layer3:10.2 cm ; 2017 pixel layer4: 16.0 cm)
  desc.add<double>("lxyWRTbsCUT",  0.); // cm
  desc.addUntracked<bool>("debug",false);
  descriptions.add("VertexCompositeCandidateCollectionSelector",desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(VertexCompositeCandidateCollectionSelector);
