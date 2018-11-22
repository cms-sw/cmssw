// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      IsFromLostTrackMapProducer
//
/**\class IsFromLostTrackMapProducer IsFromLostTrackMapProducer.cc PhysicsTools/NanoAOD/plugins/IsFromLostTrackMapProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Maria Giulia Ratti (ETHZ) [mratti]
//         Created:  Thu, 22 Nov 2018 12:34:48 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h" // not clear if this is needed

//
// class declaration
//

template <typename T>
class IsFromLostTrackMapProducer : public edm::global::EDProducer<> {
   public:
      explicit IsFromLostTrackMapProducer(const edm::ParameterSet& iConfig):
        //srcJet_(consumes<edm::View<pat::Jet>>(iConfig.getParameter<edm::InputTag>("srcJet"))) // collection of isotracks here
        // here things you want to read from the miniAOD or even from the nanoAOD
        srcIsoTracks_(consumes<std::vector<pat::IsolatedTrack>>(iConfig.getParameter<edm::InputTag>("srcIsoTracks"))), // final isolated tracks
        pc_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedPFCandidates"))), // pf candidates
        lt_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("lostTracks"))) // lost tracks
      {
        produces<edm::ValueMap<bool>>("isFromLostTrack"); // name of the value map that I want to actually produce
      }
      ~IsFromLostTrackMapProducer() override {};

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;


      // ----------member data ---------------------------
      edm::EDGetTokenT<edm::View<pat::IsolatedTrack>> srcIsoTracks_; // why only view in this case ?
      edm::EDGetTokenT<pat::PackedCandidateCollection> pc_;
      edm::EDGetTokenT<pat::PackedCandidateCollection> lt_;
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// member functions
//

// ------------ method called to produce the data  ------------
template <typename T>
void
IsFromLostTrackMapProducer<T>::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{

   // isolated tracks
   edm::Handle<edm::View<pat::IsolatedTrack>> srcIsoTracks;
   iEvent.getByToken(srcIsoTracks_, srcIsoTracks);

   // packedPFCandidate collection - not used at the moment
   edm::Handle<pat::PackedCandidateCollection> pc_handle;
   iEvent.getByToken( pc_, pc_handle );
   const pat::PackedCandidateCollection *pc = pc_handle.product();

   // lostTracks collection
   edm::Handle<pat::PackedCandidateCollection> lt_handle;
   iEvent.getByToken( lt_, lt_handle );
   const pat::PackedCandidateCollection *lt = lt_handle.product();


   // the map cannot be filled straight away, so create an intermediate vector
   unsigned int Nit = srcIsoTracks->size();
   std::vector<bool> v_isFromLostTrack(Nit,-1);

   // now I want to loop over the isolated tracks and get the reference to the pf candidate
   //for (const auto & obj : *srcIsoTracks) {
   for (unsigned int iit=0; iit<Nit; iit++){

     auto isotrack = srcIsoTracks->ptrAt(iit);

     pat::PackedCandidateRef pcref = isotrack->packedCandRef(); // this is either the reference to the pf candidate or to the lost track

     bool isFromLostTrack  = (pcref.isNonnull() && pcref.id()==lt_handle.id());

     std::cout << " isFromLostTrack " << isFromLostTrack << std::endl;

     v_isFromLostTrack[iit] = isFromLostTrack;
     // bool isInPackedCands = (pcref.isNonnull() && pcref.id()==pc_h.id() && pfCand.charge()!=0);
     // isInPackedCands should not be needed.
     // logic in PATIsolatedTrackProducer is such that requiring (isPFcand && !isFromLostTrack) is enough to fulfill isInPackedCands

   }


   std::unique_ptr<edm::ValueMap<float>> vm_isFromLostTrack(new edm::ValueMap<float>());
   edm::ValueMap<float>::Filler filler(*vm_isFromLostTrack);
   filler.insert(srcIsoTracks,v_isFromLostTrack.begin(),v_isFromLostTrack.end());
   filler.fill();
   iEvent.put(std::move(vm_isFromLostTrack),"isFromLostTrack");



}



// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
template <typename T>
void
IsFromLostTrackMapProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcIsoTracks")->setComment("isolated track input collection");
  desc.add<edm::InputTag>("pc")->setComment("packed PF Candidates collection ");
  desc.add<edm::InputTag>("lt")->setComment("lost tracks collection");

  std::string modname;
  modname="isFromLostTrack map producer ";
  descriptions.add(modname,desc);

}

//define this as a plug-in
//DEFINE_FWK_MODULE(IsFromLostTrackMapProducer);
