// -*- C++ -*-
//
// Package:    RecoLocalCalo/HcalRecProducers
// Class:      HBHEPlan1Combiner
// 
/**\class HBHEPlan1Combiner HBHEPlan1Combiner.cc RecoLocalCalo/HcalRecProducers/plugins/HBHEPlan1Combiner.cc

 Description: rechit combiner module for the "Plan 1" scenario

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Sun, 29 Jan 2017 16:05:31 GMT
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/HcalRecHit/interface/HBHERecHitAuxSetter.h"
#include "DataFormats/HcalRecHit/interface/CaloRecHitAuxSetter.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/parsePlan1RechitCombiner.h"

#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

//
// class declaration
//
class HBHEPlan1Combiner : public edm::stream::EDProducer<>
{
public:
    explicit HBHEPlan1Combiner(const edm::ParameterSet&);
    ~HBHEPlan1Combiner() override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    void produce(edm::Event&, const edm::EventSetup&) override;

    // ----------member data ---------------------------

    // Configuration parameters
    edm::EDGetTokenT<HBHERecHitCollection> tok_rechits_;
    bool ignorePlan1Topology_;
    bool usePlan1Mode_;

    // Other members
    std::unique_ptr<AbsPlan1RechitCombiner> combiner_;
};

//
// constructors and destructor
//
HBHEPlan1Combiner::HBHEPlan1Combiner(const edm::ParameterSet& conf)
    : ignorePlan1Topology_(conf.getParameter<bool>("ignorePlan1Topology")),
      usePlan1Mode_(conf.getParameter<bool>("usePlan1Mode")),
      combiner_(parsePlan1RechitCombiner(conf.getParameter<edm::ParameterSet>("algorithm")))
{
    // Check that the rechit recombination algorithm has been successfully configured
    if (!combiner_.get())
        throw cms::Exception("HBHEPlan1BadConfig")
            << "Invalid Plan1RechitCombiner algorithm configuration"
            << std::endl;

    // Consumes and produces statements
    tok_rechits_ = consumes<HBHERecHitCollection>(conf.getParameter<edm::InputTag>("hbheInput"));
    produces<HBHERecHitCollection>();
}


HBHEPlan1Combiner::~HBHEPlan1Combiner()
{ 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HBHEPlan1Combiner::produce(edm::Event& iEvent, const edm::EventSetup& eventSetup)
{
    using namespace edm;

    // Get the Hcal topology
    ESHandle<HcalTopology> htopo;
    eventSetup.get<HcalRecNumberingRecord>().get(htopo);
    combiner_->setTopo(htopo.product());

    // Are we using "Plan 1" geometry?
    const bool plan1Mode = ignorePlan1Topology_ ? usePlan1Mode_
                                                : htopo->withSpecialRBXHBHE();

    // Find the input rechit collection
    Handle<HBHERecHitCollection> inputRechits;
    iEvent.getByToken(tok_rechits_, inputRechits);

    // Create a new output collections
    std::unique_ptr<HBHERecHitCollection> outputRechits = std::make_unique<HBHERecHitCollection>();
    outputRechits->reserve(inputRechits->size());

    // Iterate over the input collection. Copy QIE8 rechits directly into
    // the output collection and prepare to combine QIE11 rechits.
    combiner_->clear();
    for (typename HBHERecHitCollection::const_iterator it = inputRechits->begin();
         it != inputRechits->end(); ++it)
    {
        // If the rechit has TDC time info, it corresponds to QIE11
        if (plan1Mode && CaloRecHitAuxSetter::getBit(
                it->auxPhase1(), HBHERecHitAuxSetter::OFF_TDC_TIME))
            combiner_->add(*it);
        else
            outputRechits->push_back(*it);
    }

    // Combine QIE11 rechits and fill the output collection
    combiner_->combine(&*outputRechits);

    // Put the output collection into the Event
    iEvent.put(std::move(outputRechits));
}

#define add_param_set(name) /**/       \
    edm::ParameterSetDescription name; \
    name.setAllowAnything();           \
    desc.add<edm::ParameterSetDescription>(#name, name)

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HBHEPlan1Combiner::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("hbheInput");
    desc.add<bool>("ignorePlan1Topology");
    desc.add<bool>("usePlan1Mode");

    add_param_set(algorithm);

    descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HBHEPlan1Combiner);
