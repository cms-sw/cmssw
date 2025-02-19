#ifndef HLTJetCollectionsForLeptonPlusJets_h
#define HLTJetCollectionsForLeptonPlusJets_h

/** \class HLTJetCollectionsForLeptonPlusJets
 *
 *
 *  This class is an EDProducer implementing an HLT
 *  trigger for lepton and jet objects, cutting on
 *  variables relating to the jet 4-momentum representation.
 *  The producer checks for overlaps between leptons and jets and if a
 *  combination of one lepton + jets cleaned against this leptons satisfy the cuts.
 *  These jets are then added to a cleaned jet collection which is put into the event.
 *
 *  $Date: 2012/02/06 15:09:21 $
 *  $Revision: 1.4 $
 *
 *  \author Lukasz Kreczko
 *
 */


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

template <typename jetType> class HLTJetCollectionsForLeptonPlusJets: public edm::EDProducer {
  public:
    explicit HLTJetCollectionsForLeptonPlusJets(const edm::ParameterSet&);
    ~HLTJetCollectionsForLeptonPlusJets();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  private:
    virtual void produce(edm::Event&, const edm::EventSetup&);

    edm::InputTag hltLeptonTag;
    edm::InputTag sourceJetTag;

    double minDeltaR_; //min dR for jets and leptons not to match

    // ----------member data ---------------------------
};
#endif //HLTJetCollectionsForLeptonPlusJets_h
