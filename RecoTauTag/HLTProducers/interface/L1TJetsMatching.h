// -*- C++ -*-
//
// Package:    RecoTauTag/HLTProducers
// Class:      L1TJetsMatching
// 
/**\class L1TJetsMatching L1TJetsMatching.h 
 RecoTauTag/HLTProducers/interface/L1TJetsMatching.h
 Description: 
 Matching L1 to PF/Calo Jets. Used for HLT_VBF paths.
	*Matches PF/Calo Jets to L1 jets from the dedicated seed
	*Adds selection criteria to the leading/subleading jets as well as the maximum dijet mass
	*Separates collections of PF/Calo jets into two categories
 
 
*/
//
// Original Author:  Vukasin Milosevic
//         Created:  Thu, 01 Jun 2017 17:23:00 GMT
//
//




#ifndef RecoTauTag_HLTProducers_L1TJetsMatching_h
#define RecoTauTag_HLTProducers_L1TJetsMatching_h

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
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <map>
#include <vector>

template< typename T>
class L1TJetsMatching: public edm::global::EDProducer<> {
 public:
  explicit L1TJetsMatching(const edm::ParameterSet&);
  ~L1TJetsMatching() override;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
    
  const edm::EDGetTokenT<std::vector<T>> jetSrc_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> jetTrigger_;
  const double pt1Min_;
  const double pt2Min_;
  const double mjjMin_;
  const double matchingR_;
  const double matchingR2_;
  };
    //
    // class decleration
    //
    using namespace reco   ;
    using namespace std    ;
    using namespace edm    ;
    using namespace trigger;
    
    
    template< typename T>
    std::pair<std::vector<T>,std::vector<T>> categorise(const std::vector<T>& pfMatchedJets, double pt1, double pt2, double Mjj)
    {
        std::pair<std::vector<T>,std::vector<T>> output;
        unsigned int i1 = 0;
        unsigned int i2 = 0;
        double mjj = 0;
        if (pfMatchedJets.size()>1){
            for (unsigned int i = 0; i < pfMatchedJets.size()-1; i++){
                
                const T &  myJet1 = (pfMatchedJets)[i];
                
                for (unsigned int j = i+1; j < pfMatchedJets.size(); j++)
                {
                    const T &  myJet2 = (pfMatchedJets)[j];
                    
                    const double mjj_test = (myJet1.p4()+myJet2.p4()).M();
                    
                    if (mjj_test > mjj){
                        
                        mjj =mjj_test;
                        i1 = i;
                        i2 = j;
                    }
                }
            }
            
            const T &  myJet1 = (pfMatchedJets)[i1];
            const T &  myJet2 = (pfMatchedJets)[i2];
            
            if ((mjj > Mjj) && (myJet1.pt() >= pt1) && (myJet2.pt() > pt2) )
            {
                
                output.first.push_back(myJet1);
                output.first.push_back(myJet2);
                
            }
            
            if ((mjj > Mjj) && (myJet1.pt() < pt1) && (myJet1.pt() > pt2) && (myJet2.pt() > pt2))
            {
                
                const T &  myJetTest = (pfMatchedJets)[0];
                if (myJetTest.pt()>pt1){
                    output.second.push_back(myJet1);
                    output.second.push_back(myJet2);
                    output.second.push_back(myJetTest);
                    
                }
            }
            
        }
        
        return output;
        
    }
    template< typename T>
    L1TJetsMatching<T>::L1TJetsMatching(const edm::ParameterSet& iConfig):
    jetSrc_    ( consumes<std::vector<T>>                     (iConfig.getParameter<InputTag>("JetSrc"      ) ) ),
    jetTrigger_( consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<InputTag>("L1JetTrigger") ) ),
    pt1Min_   ( iConfig.getParameter<double>("pt1Min")),
    pt2Min_   ( iConfig.getParameter<double>("pt2Min")),
    mjjMin_   ( iConfig.getParameter<double>("mjjMin")),
    matchingR_ ( iConfig.getParameter<double>("matchingR")),
    matchingR2_ ( matchingR_*matchingR_ )
    {
        produces<std::vector<T>>("TwoJets");
        produces<std::vector<T>>("ThreeJets");
        
    }
    template< typename T>
    L1TJetsMatching<T>::~L1TJetsMatching(){ }
    
    template< typename T>
    void L1TJetsMatching<T>::produce(edm::StreamID iSId, edm::Event& iEvent, const edm::EventSetup& iES) const
    {
        
        unique_ptr<std::vector<T>> pfMatchedJets(new std::vector<T>);
        std::pair<std::vector<T>,std::vector<T>> output;
        
        
        
        // Getting HLT jets to be matched
        edm::Handle<std::vector<T> > pfJets;
        iEvent.getByToken( jetSrc_, pfJets );
        
        edm::Handle<trigger::TriggerFilterObjectWithRefs> l1TriggeredJets;
        iEvent.getByToken(jetTrigger_,l1TriggeredJets);
        
        //l1t::TauVectorRef jetCandRefVec;
        l1t::JetVectorRef jetCandRefVec;
        l1TriggeredJets->getObjects( trigger::TriggerL1Jet,jetCandRefVec);
        
        math::XYZPoint a(0.,0.,0.);
        
        //std::cout<<"PFsize= "<<pfJets->size()<<endl<<" L1size= "<<jetCandRefVec.size()<<std::endl;
        for(unsigned int iJet = 0; iJet < pfJets->size(); iJet++){
            const T &  myJet = (*pfJets)[iJet];
            for(unsigned int iL1Jet = 0; iL1Jet < jetCandRefVec.size(); iL1Jet++){
                // Find the relative L2pfJets, to see if it has been reconstructed
                //  if ((iJet<3) && (iL1Jet==0))  std::cout<<myJet.p4().Pt()<<" ";
                if ((reco::deltaR2(myJet.p4(), jetCandRefVec[iL1Jet]->p4()) < matchingR2_ ) && (myJet.pt()>pt2Min_)) {
                    pfMatchedJets->push_back(myJet);
                    break;
                }
            }
        }
        
        output= categorise(*pfMatchedJets,pt1Min_,pt2Min_, mjjMin_);
        unique_ptr<std::vector<T>> output1(new std::vector<T>(output.first));
        unique_ptr<std::vector<T>> output2(new std::vector<T>(output.second));
        
        iEvent.put(std::move(output1),"TwoJets");
        iEvent.put(std::move(output2),"ThreeJets");
        
    }
    template< typename T>
     void L1TJetsMatching<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
     {
     edm::ParameterSetDescription desc;
     desc.add<edm::InputTag>("L1JetTrigger", edm::InputTag("hltL1DiJetVBF"))->setComment("Name of trigger filter"    );
     desc.add<edm::InputTag>("JetSrc"      , edm::InputTag("hltAK4PFJetsTightIDCorrected"))->setComment("Input collection of PFJets");
     desc.add<double>       ("pt1Min",95.0)->setComment("Minimal pT1 of PFJets to match");
     desc.add<double>       ("pt2Min",35.0)->setComment("Minimal pT2 of PFJets to match");
     desc.add<double>       ("mjjMin",650.0)->setComment("Minimal mjj of matched PFjets");
     desc.add<double>       ("matchingR",0.5)->setComment("dR value used for matching");
     descriptions.setComment("This module produces collection of PFJetss matched to L1 Taus / Jets passing a HLT filter (Only p4 and vertex of returned PFJetss are set).");
     descriptions.add(defaultModuleLabel<L1TJetsMatching<T>>(), desc);
     }
    


#endif
