#ifndef HLTJetCollectionsForElePlusJets_h
#define HLTJetCollectionsForElePlusJets_h

/** \class HLTJetCollectionsForElePlusJets
 *
 *
 *  This class is an EDProducer implementing an HLT
 *  trigger for electron and jet objects, cutting on
 *  variables relating to the jet 4-momentum representation.
 *  The producer checks for overlaps between electrons and jets and if a
 *  combination of one electron + jets cleaned against this electrons satisfy the cuts.
 *  These jets are then added to a cleaned jet collection which is put into the event.
 *
 *  $Date: 2012/02/12 10:24:17 $
 *  $Revision: 1.2 $
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

template<typename T>
class HLTJetCollectionsForElePlusJets: public edm::EDProducer {
  public:
    explicit HLTJetCollectionsForElePlusJets(const edm::ParameterSet&);
    ~HLTJetCollectionsForElePlusJets();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  private:
    virtual void produce(edm::Event&, const edm::EventSetup&);

    edm::InputTag hltElectronTag;
    edm::InputTag sourceJetTag;

//    double minJetPt_; // jet pt threshold in GeV
//    double maxAbsJetEta_; // jet |eta| range
//    unsigned int minNJets_; // number of required jets passing cuts after cleaning

    double minDeltaR_; //min dR for jets and electrons not to match
    
//    double minSoftJetPt_; // jet pt threshold for the soft jet in the VBF pair
//    double minDeltaEta_; // pseudorapidity separation for the VBF pair

    // ----------member data ---------------------------
};
#endif //HLTJetCollectionsForElePlusJets_h
