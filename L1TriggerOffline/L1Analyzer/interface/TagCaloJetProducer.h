#ifndef TagCaloJetProducer_h
#define TagCaloJetProducer_h
// -*- C++ -*-
//
// Class:      TagCaloJetProducer
// 
/**\class TagCaloJetProducer 

 Description: Make collection of tagged jets

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alex Tapper
//         Created:  Fri Jan 19 14:30:35 CET 2007
// $Id: TagCaloJetProducer.h,v 1.4 2007/05/10 16:49:29 tapper Exp $
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

//
// class declaration
//

class TagCaloJetProducer : public edm::EDProducer {
   public:
      explicit TagCaloJetProducer(const edm::ParameterSet&);
      ~TagCaloJetProducer();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      
      edm::InputTag source_; 
      double disMin_;
};

TagCaloJetProducer::TagCaloJetProducer(const edm::ParameterSet& iConfig):
  source_(iConfig.getParameter<edm::InputTag>("src")),
  disMin_(iConfig.getParameter<double>("disMin"))
{
   produces<reco::CaloJetCollection>();
}

TagCaloJetProducer::~TagCaloJetProducer()
{
}

void TagCaloJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<reco::JetTagCollection> tags;
  iEvent.getByLabel(source_, tags);

  std::auto_ptr<reco::CaloJetCollection> cand(new reco::CaloJetCollection);

  for (reco::JetTagCollection::const_iterator t=tags->begin(); t!=tags->end(); t++){
      if (t->discriminator()>disMin_) {
        const reco::CaloJet j = *(t->jet().castTo<reco::CaloJetRef>());
        //        const reco::CaloJet* j = dynamic_cast<const reco::CaloJet*>(&(t->jet()));
        cand->push_back(j);
    }
  }

  iEvent.put(cand);
}

#endif

