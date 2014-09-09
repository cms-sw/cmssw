// -*- C++ -*-
//
// Package:    JetMETCorrections/Modules
// Class:      ChainedJetCorrectorProducer
// 
/**\class ChainedJetCorrectorProducer ChainedJetCorrectorProducer.cc JetMETCorrections/Modules/plugins/ChainedJetCorrectorProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Christopher Jones
//         Created:  Tue, 02 Sep 2014 18:11:02 GMT
//
//


// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"

#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrectorImpl.h"


//
// class declaration
//
namespace {
  class ChainedJetCorrectorImpl : public reco::JetCorrectorImpl {
  public:
    ChainedJetCorrectorImpl(std::vector<reco::JetCorrector const*> correctors):
      jetCorrectors_(std::move(correctors)) {}

    virtual double correction (const LorentzVector& fJet) const override;

    /// apply correction using Jet information only
    virtual double correction (const reco::Jet& fJet) const override;

    /// apply correction using Ref
    virtual double correction (const reco::Jet& fJet,
			       const edm::RefToBase<reco::Jet>& fJetRef) const override ;

    /// if correction needs the jet reference
    virtual bool refRequired () const override;
  
  private:
    std::vector<reco::JetCorrector const*> jetCorrectors_;
  };

  double ChainedJetCorrectorImpl::correction (const LorentzVector& fJet) const
  {
    LorentzVector jet = fJet;
    double result = 1;
    for (auto cor: jetCorrectors_) {
      double scale = cor->correction (jet);
      jet *= scale;
      result *= scale;
    }
    return result;
  }
  
  /// apply correction using Jet information only
  double ChainedJetCorrectorImpl::correction (const reco::Jet& fJet) const
  {
    std::unique_ptr<reco::Jet> jet (dynamic_cast<reco::Jet*> (fJet.clone ()));
    double result = 1;
    for (auto cor: jetCorrectors_) {
      double scale = cor->correction (*jet);
      jet->scaleEnergy (scale);
      result *= scale;
    }
    return result;
  }

  /// apply correction using reference to the raw jet
  double ChainedJetCorrectorImpl::correction (const reco::Jet& fJet,
					      const edm::RefToBase<reco::Jet>& fJetRef) const
  {
    std::unique_ptr<reco::Jet> jet (dynamic_cast<reco::Jet*> (fJet.clone ()));
    double result = 1;
    for (auto cor: jetCorrectors_) {
      double scale = cor->correction (*jet, fJetRef);
      jet->scaleEnergy (scale);
      result *= scale;
    }
    return result;
  }

  /// if correction needs jet reference
  bool ChainedJetCorrectorImpl::refRequired () const
  {
    for (auto cor: jetCorrectors_) {
      if (cor->refRequired ()) return true;
    }
    return false;
  }

}


class ChainedJetCorrectorProducer : public edm::stream::EDProducer<> {
   public:
      explicit ChainedJetCorrectorProducer(const edm::ParameterSet&);

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      
      // ----------member data ---------------------------
  std::vector<edm::EDGetTokenT<reco::JetCorrector>> correctorTokens_;
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
ChainedJetCorrectorProducer::ChainedJetCorrectorProducer(const edm::ParameterSet& iConfig)
{
   //register your products
  produces<reco::JetCorrector>();

  auto const& tags = iConfig.getParameter<std::vector<edm::InputTag>>("correctors");
  correctorTokens_.reserve(tags.size());

  for(auto const& tag : tags) {
    correctorTokens_.emplace_back( consumes<reco::JetCorrector>(tag));
  }
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ChainedJetCorrectorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   std::vector<reco::JetCorrector const*> correctors;
   correctors.reserve(correctorTokens_.size());

   for(auto const& token: correctorTokens_) {
     edm::Handle<reco::JetCorrector> hCorrector;
     iEvent.getByToken(token,hCorrector);
     correctors.emplace_back(&(*hCorrector));
   }

   std::auto_ptr<reco::JetCorrector> pCorr( new reco::JetCorrector( std::unique_ptr<reco::JetCorrectorImpl>(new ChainedJetCorrectorImpl(std::move(correctors)) )));
   iEvent.put(pCorr);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
ChainedJetCorrectorProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("correctors");
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ChainedJetCorrectorProducer);
