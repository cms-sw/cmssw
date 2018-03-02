#ifndef RecoTauTag_HLTProducers_L1TJetsMatchingOnly_h
#define RecoTauTag_HLTProducers_L1TJetsMatchingOnly_h

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <map>
#include <vector>

template<typename T>
class L1TJetsMatchingOnly: public edm::global::EDProducer<> {
    public:
        explicit L1TJetsMatchingOnly(const edm::ParameterSet&);
        ~L1TJetsMatchingOnly() override;
        void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
        static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    private:
        const edm::EDGetTokenT<std::vector<T>> jetSrc_;
        const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> jetTrigger_;
        const double matchingR2_;
};
    //
    // class declaration
    //
template<typename T>
L1TJetsMatchingOnly<T>::L1TJetsMatchingOnly(const edm::ParameterSet& iConfig):
    jetSrc_    ( consumes<std::vector<T>>                     (iConfig.getParameter<edm::InputTag>("JetSrc"      ) ) ),
    jetTrigger_( consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<edm::InputTag>("L1JetTrigger") ) ),
    matchingR2_ ( iConfig.getParameter<double>("matchingR")*iConfig.getParameter<double>("matchingR") )
{
    produces<std::vector<T>>();
    
}

template< typename T>
L1TJetsMatchingOnly<T>::~L1TJetsMatchingOnly(){ }

template< typename T>
void L1TJetsMatchingOnly<T>::produce(edm::StreamID iSId, edm::Event& iEvent, const edm::EventSetup& iES) const
{
    std::unique_ptr<std::vector<T>> L1MatchedJets(new std::vector<T>);
    
    // Getting HLT jets to be matched
    edm::Handle<std::vector<T> > pfJets;
    iEvent.getByToken( jetSrc_, pfJets );
    
    edm::Handle<trigger::TriggerFilterObjectWithRefs> l1TriggeredJets;
    iEvent.getByToken(jetTrigger_, l1TriggeredJets);
    
    l1t::JetVectorRef jetCandRefVec;
    l1TriggeredJets->getObjects(trigger::TriggerL1Jet, jetCandRefVec);
    
    for(unsigned int iJet = 0; iJet < pfJets->size(); iJet++){
        const T & myJet = (*pfJets)[iJet];
        for(unsigned int iL1Jet = 0; iL1Jet < jetCandRefVec.size(); iL1Jet++){
            if (reco::deltaR2(myJet.p4(), jetCandRefVec[iL1Jet]->p4()) < matchingR2_ ){
                L1MatchedJets->push_back(myJet);
                break;
            }
        }
    }
    iEvent.put(std::move(L1MatchedJets));
    
}
template< typename T>
 void L1TJetsMatchingOnly<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
 {
 edm::ParameterSetDescription desc;
 desc.add<edm::InputTag>("L1JetTrigger", edm::InputTag("hltL1VBFDiJetOR"))->setComment("Name of trigger filter"    );
 desc.add<edm::InputTag>("JetSrc"      , edm::InputTag("hltAK4PFJetsLooseIDCorrected"))->setComment("Input collection of PFJets");
 desc.add<double>       ("matchingR",0.5)->setComment("dR value used for matching");
 descriptions.setComment("This module produces a collection of PFJets matched to L1 Jets.");
 descriptions.add(defaultModuleLabel<L1TJetsMatchingOnly<T>>(), desc);
 }
#endif
