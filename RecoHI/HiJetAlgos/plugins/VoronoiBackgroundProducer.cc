// -*- C++ -*-
//
// Package:    VoronoiBackgroundProducer
// Class:      VoronoiBackgroundProducer
// 
/**\class VoronoiBackgroundProducer VoronoiBackgroundProducer.cc RecoHI/VoronoiBackgroundProducer/plugins/VoronoiBackgroundProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Tue, 09 Jul 2013 14:28:26 GMT
// $Id$
//
//


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
      virtual void beginJob();
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob();
      
      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

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
   equalizeR_(iConfig.getParameter<double>("equalizeR"))
{

   voronoi_ = new VoronoiAlgorithm(equalizeR_);
   src_ = iConfig.getParameter<edm::InputTag>("src");
   //register your products

   produces<reco::VoronoiMap>();
   //   produces<reco::VoronoiBackgroundMap>();
   //   produces<reco::PFVoronoiBackgroundMap>();

}


VoronoiBackgroundProducer::~VoronoiBackgroundProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
VoronoiBackgroundProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   voronoi_->clear();

   edm::Handle<reco::CandidateView> inputsHandle;
   iEvent.getByLabel(src_,inputsHandle);
   //   std::auto_ptr<reco::VoronoiBackgroundMap> mapout(new reco::VoronoiBackgroundMap());
   std::auto_ptr<reco::VoronoiMap> mapout(new reco::VoronoiMap());

   reco::VoronoiMap::Filler filler(*mapout);
   vvm.clear();

   //   edm::Handle<reco::PFCandidateCollection> inputsHandle;
   //   iEvent.getByLabel(src_,inputsHandle);
   //   std::auto_ptr<reco::PFVoronoiBackgroundMap> mapout(new reco::PFVoronoiBackgroundMap());

   for(unsigned int i = 0; i < inputsHandle->size(); ++i){
      reco::CandidateViewRef ref(inputsHandle,i);
      //      edm::Ref<reco::PFCandidateCollection> ref(inputsHandle,i);
      voronoi_->push_back_particle(ref->pt(),ref->eta(),ref->phi(),0);
   }

   std::vector<double> momentum_perp_subtracted = (*voronoi_);

   for(unsigned int i = 0; i < inputsHandle->size(); ++i){
      reco::CandidateViewRef ref(inputsHandle,i);
      //      edm::Ref<reco::PFCandidateCollection> ref(inputsHandle,i);

      reco::VoronoiBackground bkg(0,0,momentum_perp_subtracted[i],0,0,0);

      vvm.push_back(bkg);

   }

   filler.insert(inputsHandle,vvm.begin(),vvm.end());
   filler.fill();
   iEvent.put(mapout);
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
VoronoiBackgroundProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
VoronoiBackgroundProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
VoronoiBackgroundProducer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
VoronoiBackgroundProducer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
VoronoiBackgroundProducer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
VoronoiBackgroundProducer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
VoronoiBackgroundProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(VoronoiBackgroundProducer);
