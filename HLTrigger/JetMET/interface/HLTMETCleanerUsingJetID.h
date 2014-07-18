#ifndef HLTMETCleanerUsingJetID_h_
#define HLTMETCleanerUsingJetID_h_

/** \class  HLTMETCleanerUsingJetID
 *
 *  \brief  This creates a MET object from the difference in MET between two
 *          input jet collections.
 *  \author Jia Fu Low (Nov 2013)
 *
 *  This code creates a new MET vector defined as:
 *
 *    output MET = input MET + MET from 'good jets' - MET from 'all jets'
 *
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"


namespace edm {
   class ConfigurationDescriptions;
}

// Class declaration
class HLTMETCleanerUsingJetID : public edm::EDProducer {
  public:
    explicit HLTMETCleanerUsingJetID(const edm::ParameterSet & iConfig);
    ~HLTMETCleanerUsingJetID();

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual void produce(edm::Event& iEvent, const edm::EventSetup & iSetup);

  private:
    /// Use pt; otherwise, use et.
    bool            usePt_;

    /// Minimum pt requirement for jets
    double          minPt_;

    /// Maximum (abs) eta requirement for jets
    double          maxEta_;

    /// Input tag for the MET collection
    edm::InputTag   metLabel_;

    /// Input tag for the 'all jets' collection
    edm::InputTag   jetsLabel_;

    /// Input tag for the 'good jets' collection
    edm::InputTag   goodJetsLabel_;

    edm::EDGetTokenT<reco::CaloMETCollection> m_theMETToken;
    edm::EDGetTokenT<reco::CaloJetCollection> m_theJetToken;
    edm::EDGetTokenT<reco::CaloJetCollection> m_theGoodJetToken;
};

#endif  // HLTMETCleanerUsingJetID_h_
