/** \class HLTCaloJetIDProducer
 *
 * See header file for documentation
 *
 *  \author a Jet/MET person
 *
 */

#include "HLTrigger/JetMET/interface/HLTCaloJetIDProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"


// Constructor
HLTCaloJetIDProducer::HLTCaloJetIDProducer(const edm::ParameterSet& iConfig) :
  min_N90_    (iConfig.getParameter<int>("min_N90")),
  min_N90hits_(iConfig.getParameter<int>("min_N90hits")),
  min_EMF_    (iConfig.getParameter<double>("min_EMF")),
  max_EMF_    (iConfig.getParameter<double>("max_EMF")),
  inputTag_   (iConfig.getParameter<edm::InputTag>("jetsInput")),
  jetIDParams_(iConfig.getParameter<edm::ParameterSet>("JetIDParams")),
  jetIDHelper_(jetIDParams_,consumesCollector()) {
    m_theCaloJetToken = consumes<reco::CaloJetCollection>(inputTag_);

    // Register the products
    produces<reco::CaloJetCollection>();
}

// Destructor
HLTCaloJetIDProducer::~HLTCaloJetIDProducer() {}

// Fill descriptions
void HLTCaloJetIDProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<int>("min_N90", -2);
    desc.add<int>("min_N90hits", 2);
    desc.add<double>("min_EMF", 1e-6);
    desc.add<double>("max_EMF", 999.);
    desc.add<edm::InputTag>("jetsInput", edm::InputTag("hltAntiKT4CaloJets"));

    edm::ParameterSetDescription descNested;
    descNested.add<bool>("useRecHits", true);
    descNested.add<edm::InputTag>("hbheRecHitsColl", edm::InputTag("hltHbhereco"));
    descNested.add<edm::InputTag>("hoRecHitsColl", edm::InputTag("hltHoreco"));
    descNested.add<edm::InputTag>("hfRecHitsColl", edm::InputTag("hltHfreco"));
    descNested.add<edm::InputTag>("ebRecHitsColl", edm::InputTag("hltEcalRecHit","EcalRecHitsEB"));
    descNested.add<edm::InputTag>("eeRecHitsColl", edm::InputTag("hltEcalRecHit","EcalRecHitsEE"));
    desc.add<edm::ParameterSetDescription>("JetIDParams", descNested);

    descriptions.add("hltCaloJetIDProducer", desc);
}

// Produce the products
void HLTCaloJetIDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    // Create a pointer to the products
    std::auto_ptr<reco::CaloJetCollection> result (new reco::CaloJetCollection());

    edm::Handle<reco::CaloJetCollection> calojets;
    iEvent.getByToken(m_theCaloJetToken, calojets);

    for (reco::CaloJetCollection::const_iterator j = calojets->begin(); j != calojets->end(); ++j) {
        bool pass = false;

        if (!(j->energy() > 0.))  continue;  // skip jets with zero or negative energy

        if (std::abs(j->eta()) >= 2.6) {
            pass = true;

        } else {
            if (min_N90hits_ > 0)  jetIDHelper_.calculate(iEvent, *j);
            if ((j->emEnergyFraction() >= min_EMF_) &&
                (j->emEnergyFraction() <= max_EMF_) &&
                (j->n90() >= min_N90_) &&
                ((min_N90hits_ <= 0) || (jetIDHelper_.n90Hits() >= min_N90hits_)) ) {

                pass = true;
            }
        }

        if (pass)  result->push_back(*j);
    }

    // Put the products into the Event
    iEvent.put(result);
}
