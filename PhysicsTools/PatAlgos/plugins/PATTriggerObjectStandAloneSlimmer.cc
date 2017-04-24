// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      pat::PATTriggerObjectStandAloneSlimmer
//
//
/**
  \class    pat::PATTriggerObjectStandAloneSlimmer PATTriggerObjectStandAloneSlimmer.h "PhysicsTools/PatAlgos/plugins/PATTriggerObjectStandAloneSlimmer.cc"
  \brief    Packs filter labels and/or 4-vectors of a pat::TriggerObjectStandAloneCollection

  \author   Giovanni Petrucciani
*/


#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include <set>

namespace pat {
  
  class PATTriggerObjectStandAloneSlimmer : public edm::global::EDProducer<> {
    
  public:
    
    explicit PATTriggerObjectStandAloneSlimmer( const edm::ParameterSet & iConfig );
    ~PATTriggerObjectStandAloneSlimmer() {};
    
  private:
    
    virtual void produce(edm::StreamID, edm::Event & iEvent, const edm::EventSetup& iSetup) const override;
    
    const edm::EDGetTokenT<TriggerObjectStandAloneCollection> srcToken_;
   
    bool packFilterLabels_, packP4_;
 
  };
  
}


using namespace pat;


PATTriggerObjectStandAloneSlimmer::PATTriggerObjectStandAloneSlimmer( const edm::ParameterSet & iConfig ) : 
    srcToken_( consumes<TriggerObjectStandAloneCollection>( iConfig.getParameter<edm::InputTag>( "src" ) ) ),
    packFilterLabels_( iConfig.getParameter<bool>("packFilterLabels") ),
    packP4_( iConfig.getParameter<bool>("packP4") )
{
    produces<TriggerObjectStandAloneCollection>();
    if (packFilterLabels_) {
        produces<std::vector<std::string>>("filterLabels");
    }
}

void PATTriggerObjectStandAloneSlimmer::produce( edm::StreamID, edm::Event & iEvent, const edm::EventSetup& iSetup) const
{
    edm::Handle<TriggerObjectStandAloneCollection> src;
    iEvent.getByToken( srcToken_, src );

    auto slimmed = std::make_unique<TriggerObjectStandAloneCollection>(*src);

    if (packFilterLabels_) {
        std::set<std::string> allLabels;
        for (const auto & obj : *slimmed) {
            for (const std::string & label : obj.filterLabels()) {
                allLabels.insert(label);
            }
        }
        auto newlabels = std::make_unique<std::vector<std::string>>();
        newlabels->reserve(allLabels.size());
        newlabels->insert(newlabels->end(), allLabels.begin(), allLabels.end());
        for (TriggerObjectStandAlone & obj : *slimmed) {
            obj.packFilterLabels(*newlabels);
        }
        iEvent.put(std::move(newlabels), "filterLabels");

    }
    if (packP4_) {
        for (TriggerObjectStandAlone & obj : *slimmed) {
            obj.packP4();
        }
    }
    iEvent.put(std::move(slimmed) );
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PATTriggerObjectStandAloneSlimmer );
