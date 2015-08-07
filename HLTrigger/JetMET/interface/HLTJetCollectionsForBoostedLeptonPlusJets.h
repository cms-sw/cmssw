#ifndef HLTJetCollectionsForBoostedLeptonPlusJets_h
#define HLTJetCollectionsForBoostedLeptonPlusJets_h

/** \class HLTJetCollectionsForBoostedLeptonPlusJets
 *
 *
 *  This class is an EDProducer implementing an HLT
 *  trigger for lepton and jet objects, cutting on
 *  variables relating to the jet 4-momentum representation.
 *  The producer checks for overlaps between leptons and jets and if a
 *  combination of one lepton + jets cleaned against this leptons satisfy the cuts.
 *  These jets are then added to a cleaned jet collection which is put into the event.
 *
 *
 *  \author Lukasz Kreczko
 *
 */


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/JetReco/interface/PFJetCollection.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

template <typename jetType> class HLTJetCollectionsForBoostedLeptonPlusJets: public edm::stream::EDProducer<> {
  public:
    explicit HLTJetCollectionsForBoostedLeptonPlusJets(const edm::ParameterSet&);
    ~HLTJetCollectionsForBoostedLeptonPlusJets();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  private:
    virtual void produce(edm::Event&, const edm::EventSetup&);

    edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> m_theLeptonToken;
    edm::EDGetTokenT<std::vector<jetType>> m_theJetToken;
    edm::InputTag hltLeptonTag;
    edm::InputTag sourceJetTag;

    double minDeltaR_; //min dR to consider cleaning

    // ----------member data ---------------------------
};
#endif //HLTJetCollectionsForBoostedLeptonPlusJets_h
