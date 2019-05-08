// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      GenJetTauTaggerProducer
// 
/**\class GenJetTauTaggerProducer GenJetTauTaggerProducer.cc PhysicsTools/NanoAOD/plugins/GenJetTauTaggerProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Elisabetta Manca
//         Created:  Wed, 08 May 2019 13:09:28 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Common/interface/ValueMap.h"


//
// class declaration
//

class GenJetTauTaggerProducer : public edm::stream::EDProducer<> {
   public:
      explicit GenJetTauTaggerProducer(const edm::ParameterSet &iConfig):
        src_(consumes<std::vector<reco::GenJet> >(iConfig.getParameter<edm::InputTag>("src")))
        {
          produces<edm::ValueMap<int>>();
        } 
      ~GenJetTauTaggerProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      edm::EDGetTokenT<std::vector<reco::GenJet> > src_;

};



GenJetTauTaggerProducer::~GenJetTauTaggerProducer()
{

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GenJetTauTaggerProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  edm::Handle<reco::GenJetCollection> jets;
  iEvent.getByToken(src_, jets);

  std::vector<int> tags;

  for(auto jet = jets->begin(); jet != jets->end(); ++jet){

    int found = 0;
    for(auto cand : jet->getJetConstituentsQuick()){
      if( abs(cand->pdgId()) == 15) found = 1;
      tags.push_back(found);
    }
  }

  std::unique_ptr<edm::ValueMap<int>> tagsV(new edm::ValueMap<int>());
  edm::ValueMap<int>::Filler fillerCorr(*tagsV);
  fillerCorr.insert(jets,tags.begin(),tags.end());
  fillerCorr.fill();
  iEvent.put(std::move(tagsV));
}


void
GenJetTauTaggerProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GenJetTauTaggerProducer);
