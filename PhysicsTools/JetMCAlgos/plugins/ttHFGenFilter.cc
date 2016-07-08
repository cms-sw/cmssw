// -*- C++ -*-
//
// Package:    ttHFGenFilter/ttHFGenFilter
// Class:      ttHFGenFilter
//
/**\class ttHFGenFilter ttHFGenFilter.cc ttHFGenFilter/ttHFGenFilter/plugins/ttHFGenFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Andrej Saibel
//         Created:  Tue, 05 Jul 2016 09:36:09 GMT
//
// TODO Description


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/InputTag.h"


//
// class declaration
//

class ttHFGenFilter : public edm::stream::EDFilter<> {
   public:
      explicit ttHFGenFilter(const edm::ParameterSet&);
      ~ttHFGenFilter();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginStream(edm::StreamID) override;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;

      virtual bool HasAdditionalBHadron(const std::vector<int>&, const std::vector<int>&,const std::vector<reco::GenParticle>&);

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      const edm::EDGetTokenT<std::vector<int> > genBHadFlavourToken_;
      const edm::EDGetTokenT<std::vector<int> > genBHadFromTopWeakDecayToken_;
      const edm::EDGetTokenT<std::vector<reco::GenParticle> > genBHadPlusMothersToken_;
      const edm::EDGetTokenT<std::vector<std::vector<int> > > genBHadPlusMothersIndicesToken_;
      const edm::EDGetTokenT<std::vector<int> > genBHadIndexToken_;

      // ----------member data ---------------------------
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
ttHFGenFilter::ttHFGenFilter(const edm::ParameterSet& iConfig):
genBHadFlavourToken_(consumes<std::vector<int> >(iConfig.getParameter<edm::InputTag>("genBHadFlavour"))),
genBHadFromTopWeakDecayToken_(consumes<std::vector<int> >(iConfig.getParameter<edm::InputTag>("genBHadFromTopWeakDecay"))),
genBHadPlusMothersToken_(consumes<std::vector<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("genBHadPlusMothers"))),
genBHadPlusMothersIndicesToken_(consumes<std::vector<std::vector<int> > >(iConfig.getParameter<edm::InputTag>("genBHadPlusMothersIndices"))),
genBHadIndexToken_(consumes<std::vector<int> >(iConfig.getParameter<edm::InputTag>("genBHadIndex")))
{
  //now do what ever initialization is needed
}


ttHFGenFilter::~ttHFGenFilter()
{

   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
ttHFGenFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   //bool IsAdditonalBHadron=false;
   //get GenParticleCollection
   edm::Handle<reco::GenParticleCollection> genParticles;

   //get Information on B-Hadrons
   // the information is calculated in GenHFHadronMatcher

   edm::Handle<std::vector<int> > genBHadFlavour;
   iEvent.getByToken(genBHadFlavourToken_, genBHadFlavour);

   edm::Handle<std::vector<int> > genBHadFromTopWeakDecay;
   iEvent.getByToken(genBHadFromTopWeakDecayToken_, genBHadFromTopWeakDecay);

   edm::Handle<std::vector<reco::GenParticle> > genBHadPlusMothers;
   iEvent.getByToken(genBHadPlusMothersToken_, genBHadPlusMothers);

   edm::Handle<std::vector<std::vector<int> > > genBHadPlusMothersIndices;
   iEvent.getByToken(genBHadPlusMothersIndicesToken_, genBHadPlusMothersIndices);

   edm::Handle<std::vector<int> > genBHadIndex;
   iEvent.getByToken(genBHadIndexToken_, genBHadIndex);

   //check whether the event has additional b-hadrons not coming from top/topbar decay
   cout << "Filter wird ausgefuehrt" << endl;
   cout << "has additional b hadron " << HasAdditionalBHadron(*genBHadIndex,*genBHadFlavour,*genBHadPlusMothers) << endl;
   return HasAdditionalBHadron(*genBHadIndex,*genBHadFlavour,*genBHadPlusMothers);

   //TODO check whether the b-hadron is coming from the hard interaction or from underlying event
}

bool ttHFGenFilter::HasAdditionalBHadron(const std::vector<int>& genBHadIndex, const std::vector<int>& genBHadFlavour,const std::vector<reco::GenParticle>& genBHadPlusMothers){
  for(uint i=0; i<genBHadIndex.size();i++){
    const reco::GenParticle* bhadron = genBHadIndex[i]>=0&&genBHadIndex[i]<int(genBHadPlusMothers.size()) ? &(genBHadPlusMothers[genBHadIndex[i]]) : 0;
    int motherflav = genBHadFlavour[i];
    bool from_tth=(abs(motherflav)==6||abs(motherflav)==25); //b-hadron comes from top or higgs decay

    if(bhadron!=0&&!from_tth){
      return true;
    }
    if(i==genBHadIndex.size()-1){
      return false;
    }
  }
  return false;
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
ttHFGenFilter::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
ttHFGenFilter::endStream() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
ttHFGenFilter::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
ttHFGenFilter::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
ttHFGenFilter::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
ttHFGenFilter::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
ttHFGenFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(ttHFGenFilter);
