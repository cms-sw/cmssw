// -*- C++ -*-
//
// Package:    ttHFGenFilter/ttHFGenFilter
// Class:      ttHFGenFilter
//
/**\class ttHFGenFilter ttHFGenFilter.cc ttHFGenFilter/ttHFGenFilter/plugins/ttHFGenFilter.cc

 Description: Filters ttbar events, where additional b-hadrons come from ISR or hard process

 Implementation:
     The ttHFGenFilter is a edm::Filter, that returns true, if a tt+jets event contains a b-hadron, that does not come from top decay or when the b-hadron comes from ISR.
	The filter uses the GenHFHadronMatcher to classify, whether the b-hadron is an "additional b-hadron". This is done in the ttHFGenFilter::HasAdditionalBHadron() function.
	To classify, whether the b-hadron comes from ISR:
		first all mothers of the top-topbar pair are found
		then the mother-chain of all b-hadrons is checked if it contains at least one mother, that is also in the mother-chain of the top-topbar pair

*/
//
// Original Author:  Andrej Saibel
//         Created:  Tue, 05 Jul 2016 09:36:09 GMT
//
// VERY IMPORTANT: when running this code, you should make sure, that the GenHFHadronMatcher runs in onlyJetClusteredHadrons = cms.bool(False) mode
//                 If you want to filter all events with additional b-hadrons use: OnlyHardProcessBHadrons = cms.bool(False)


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

      virtual bool HasAdditionalBHadron(const std::vector<int>&, const std::vector<int>&,const std::vector<reco::GenParticle>&,std::vector<const reco::Candidate*>&);
      virtual bool analyzeMothersRecursive(const reco::Candidate*,std::vector<const reco::Candidate*>& AllTopMothers);
      virtual std::vector<const reco::Candidate*> GetTops(const std::vector<reco::GenParticle> &,std::vector<const reco::Candidate*>& AllTopMothers);
      virtual void FindAllTopMothers(const reco::Candidate* particle,std::vector<const reco::Candidate*>&);

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      const edm::EDGetTokenT<reco::GenParticleCollection> genParticlesToken_;
      const edm::EDGetTokenT<std::vector<int> > genBHadFlavourToken_;
      const edm::EDGetTokenT<std::vector<int> > genBHadFromTopWeakDecayToken_;
      const edm::EDGetTokenT<std::vector<reco::GenParticle> > genBHadPlusMothersToken_;
      const edm::EDGetTokenT<std::vector<std::vector<int> > > genBHadPlusMothersIndicesToken_;
      const edm::EDGetTokenT<std::vector<int> > genBHadIndexToken_;
      bool OnlyHardProcessBHadrons_;


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
genParticlesToken_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genParticles"))),
genBHadFlavourToken_(consumes<std::vector<int> >(iConfig.getParameter<edm::InputTag>("genBHadFlavour"))),
genBHadFromTopWeakDecayToken_(consumes<std::vector<int> >(iConfig.getParameter<edm::InputTag>("genBHadFromTopWeakDecay"))),
genBHadPlusMothersToken_(consumes<std::vector<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("genBHadPlusMothers"))),
genBHadPlusMothersIndicesToken_(consumes<std::vector<std::vector<int> > >(iConfig.getParameter<edm::InputTag>("genBHadPlusMothersIndices"))),
genBHadIndexToken_(consumes<std::vector<int> >(iConfig.getParameter<edm::InputTag>("genBHadIndex")))
{
  //now do what ever initialization is needed
  OnlyHardProcessBHadrons_ = iConfig.getParameter<bool> ( "OnlyHardProcessBHadrons" );
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

   //get GenParticleCollection
   edm::Handle<reco::GenParticleCollection> genParticles;
   iEvent.getByToken(genParticlesToken_, genParticles);

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

   std::vector<const reco::Candidate*> AllTopMothers;
   std::vector<const reco::Candidate*> Tops = GetTops(*genParticles,AllTopMothers);
   // std::cout << "Size of AllTopMothers = " << AllTopMothers.size() << std::endl;
   return HasAdditionalBHadron(*genBHadIndex,*genBHadFlavour,*genBHadPlusMothers,AllTopMothers);

}

bool ttHFGenFilter::HasAdditionalBHadron(const std::vector<int>& genBHadIndex, const std::vector<int>& genBHadFlavour,const std::vector<reco::GenParticle>& genBHadPlusMothers,std::vector<const reco::Candidate*>& AllTopMothers){

  for(uint i=0; i<genBHadIndex.size();i++){

    const reco::GenParticle* bhadron = genBHadIndex[i]>=0&&genBHadIndex[i]<int(genBHadPlusMothers.size()) ? &(genBHadPlusMothers[genBHadIndex[i]]) : 0;
    int motherflav = genBHadFlavour[i];
    bool from_tth=(abs(motherflav)==6||abs(motherflav)==25); //b-hadron comes from top or higgs decay
    bool fromhp=false;

  if(!OnlyHardProcessBHadrons_){
    if(bhadron!=0&&!from_tth){
      return true;
    }
  }

  if(OnlyHardProcessBHadrons_){
    if(bhadron!=0&&!from_tth){
      //std::cout << "PT: " << bhadron->pt() << " , eta: " << bhadron->eta() << std::endl;

      fromhp=analyzeMothersRecursive(bhadron,AllTopMothers);
      if(fromhp){
	return true;
      }
    }
    if(i==genBHadIndex.size()-1){
	return false;
    }
  }
}

  return false;
}

//recursive function, that loops over all chain particles  until it finds the protons
bool ttHFGenFilter::analyzeMothersRecursive(const reco::Candidate* particle,std::vector<const reco::Candidate*>& AllTopMothers){
  //std::cout << "Particle: " << particle->pdgId() << " , Status: " << particle->status() << " , numberOfMothers: " << particle->numberOfMothers() << std::endl;
  if(particle->status()>20&&particle->status()<30){ //particle comes from hardest process in event
    return true;
  }
  for(uint k=0; k<AllTopMothers.size();k++){
    if(particle==AllTopMothers[k]){ //partcile comes from  ISR

      return true;
    }
  }
    bool IsFromHardProcess=false;
  /*if(analyzeMothersRecursive(particle)){
    return true;
  }*/
    for(uint i=0;i<particle->numberOfMothers();i++){
      //std::cout << "i " <<  i << std::endl;
      const reco::Candidate* mother = particle->mother(i);
      IsFromHardProcess=analyzeMothersRecursive(mother,AllTopMothers);
      if(IsFromHardProcess){
        return true;
      }

    }

  return IsFromHardProcess;
}

std::vector< const reco::Candidate*> ttHFGenFilter::GetTops(const std::vector<reco::GenParticle>& genParticles, std::vector<const reco::Candidate*>& AllTopMothers){
  const reco::GenParticle* FirstTop = 0;
  const reco::GenParticle* FirstTopBar = 0;
  bool FoundTop = false;
  bool FoundTopBar =false;
  std::vector<const reco::GenParticle*> Tops;

  //loop over all genParticles and find  a Top and AntiTop quark
  //then find all mothers of the Tops
  //this is then used to  if the b-hadron is an inital state radiation particle

   for(reco::GenParticleCollection::const_iterator i_particle = genParticles.begin(); i_particle != genParticles.end(); ++i_particle){
       const reco::GenParticle* thisParticle = &*i_particle;
     if(thisParticle->pdgId()==6){
       FirstTop = thisParticle;
       FoundTop = true;
     }
     if(thisParticle->pdgId()==-6){
        FirstTopBar = thisParticle;
        FoundTopBar = true;
     }
     if(FoundTopBar&&FoundTop){
       //Tops.push_back(FirstTop);
       //Tops.push_back(FirstTopBar);
       //return Tops;
       break;
     }
   }
   for(uint i=0; i<FirstTop->numberOfMothers();i++){
     FindAllTopMothers(FirstTop->mother(i),AllTopMothers);
   }
   for(uint i=0; i<FirstTopBar->numberOfMothers();i++){
     FindAllTopMothers(FirstTopBar->mother(i),AllTopMothers);
   }
   return AllTopMothers;
}

// finds all mothers of "particle", that are not tops or protons

void ttHFGenFilter::FindAllTopMothers(const reco::Candidate* particle, std::vector<const reco::Candidate*>& AllTopMothers){
 // std::cout << "particle mother: " << particle->mother(0) << std::endl;
  for(uint i=0;i<particle->numberOfMothers();i++){
    if(abs(particle->mother(i)->pdgId())!=6&&particle->mother(i)->pdgId()!=2212){
      AllTopMothers.push_back(particle->mother(i));
      if(particle->mother(i)->pdgId()!=2212 || particle->mother(i)->numberOfMothers()>1){
        //std::cout << "Size of vector in loop = " << AllTopMothers.size() << std::endl;
        FindAllTopMothers(particle->mother(i),AllTopMothers);
      }
    }
  }

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
