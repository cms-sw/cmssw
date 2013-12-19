// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HeavyIonEvent/interface/VoronoiBackground.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include "RecoHI/HiJetAlgos/interface/VoronoiAlgorithm.h"

//
// class declaration
//

class VoronoiBackgroundProducer : public edm::EDProducer {
   public:
      explicit VoronoiBackgroundProducer(const edm::ParameterSet&);
      ~VoronoiBackgroundProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      
      // ----------member data ---------------------------

   edm::InputTag src_;
   VoronoiAlgorithm* voronoi_;
   double equalizeR_;
   std::vector<reco::VoronoiBackground> vvm;

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
VoronoiBackgroundProducer::VoronoiBackgroundProducer(const edm::ParameterSet& iConfig):
   voronoi_(0),
   equalizeR_(iConfig.getParameter<double>("equalizeR"))
{

   src_ = iConfig.getParameter<edm::InputTag>("src");
   //register your products

   produces<reco::VoronoiMap>();

}


VoronoiBackgroundProducer::~VoronoiBackgroundProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
VoronoiBackgroundProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   if(voronoi_ == 0){
     bool data = iEvent.isRealData();
     voronoi_ = new VoronoiAlgorithm(equalizeR_,data);
   }

   voronoi_->clear();

   edm::Handle<reco::CandidateView> inputsHandle;
   iEvent.getByLabel(src_,inputsHandle);
   //   std::auto_ptr<reco::VoronoiBackgroundMap> mapout(new reco::VoronoiBackgroundMap());
   std::auto_ptr<reco::VoronoiMap> mapout(new reco::VoronoiMap());

   reco::VoronoiMap::Filler filler(*mapout);
   vvm.clear();

   for(unsigned int i = 0; i < inputsHandle->size(); ++i){
      reco::CandidateViewRef ref(inputsHandle,i);
      voronoi_->push_back_particle(ref->pt(),ref->eta(),ref->phi(),0);
   }

   std::vector<double> momentum_perp_subtracted = (*voronoi_);

   for(unsigned int i = 0; i < inputsHandle->size(); ++i){
      reco::CandidateViewRef ref(inputsHandle,i);
      reco::VoronoiBackground bkg(0,0,momentum_perp_subtracted[i],0,0,0,0);

      vvm.push_back(bkg);

   }

   filler.insert(inputsHandle,vvm.begin(),vvm.end());
   filler.fill();
   iEvent.put(mapout);
 
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
VoronoiBackgroundProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(VoronoiBackgroundProducer);
