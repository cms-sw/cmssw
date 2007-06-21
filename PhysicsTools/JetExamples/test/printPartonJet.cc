// system include files
#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <TMath.h>
#include <TFile.h>
#include <TH1.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"

using namespace std;
using namespace reco;
using namespace edm;

class printPartonJet : public edm::EDAnalyzer {
  public:
    explicit printPartonJet(const edm::ParameterSet & );
    ~printPartonJet() {};
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
     
  private:

    edm::InputTag source_;
    edm::InputTag matched_;
    string   fOutputFileName_;   

    edm::Handle<reco::CandidateCollection> partonJets;
    edm::Handle<reco::CandMatchMap> PartonCaloMap;
};

printPartonJet::printPartonJet(const edm::ParameterSet& iConfig)
{
  source_  = iConfig.getParameter<InputTag> ("src");
  matched_ = iConfig.getParameter<InputTag> ("matched");
}

void printPartonJet::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  cout << "[printPartonJet] analysing event " << iEvent.id() << endl;
  
  try {
    iEvent.getByLabel (source_ ,partonJets);
    iEvent.getByLabel (matched_,PartonCaloMap);
  } catch(std::exception& ce) {
    cerr << "[printPartonJet] caught std::exception " << ce.what() << endl;
    return;
  }

  cout << "************************" << endl;
  cout << "* PartonJetCollection  *" << endl;
  cout << "************************" << endl;
  for( CandidateCollection::const_iterator f  = partonJets->begin();
                                           f != partonJets->end();
                                           f++) {

     printf("[printPartonJet] (pt,eta,phi) = %7.3f %6.3f %6.3f |\n",
              f->et(),
              f->eta(),
              f->phi()  );

     for( Candidate::const_iterator c  = f->begin();   
                                    c != f->end();   
                                    c ++) {  
       printf("        [Constituents] (pt,eta,phi | id |isB,isC) = %6.2f %5.2f %5.2f | %6d |\n",
               c->et(),                                                                       
               c->eta(),                                                                      
               c->phi(),  
               c->pdgId() );
     }                                                                                          
  }

  cout << "*************************" << endl;
  cout << "* Matching to Calo jets *" << endl;
  cout << "*************************" << endl;

  for( CandMatchMap::const_iterator f  = PartonCaloMap->begin();
                                    f != PartonCaloMap->end();
                                    f++) {

      const Candidate *theParton     = &*(f->key);
      const Candidate *theCaloJet  = &*(f->val);

      printf("[printParton-CaloMap] (pt,eta,phi) parton = %7.3f %6.3f %6.3f - %6d | jet = %7.3f %6.3f %6.3f |\n",
	     theParton->et(), 
	     theParton->eta(),
	     theParton->phi(), 
             theParton->pdgId(),
             theCaloJet->et(),
             theCaloJet->eta(),
             theCaloJet->phi()
	     );
  }
}

DEFINE_FWK_MODULE( printPartonJet );
