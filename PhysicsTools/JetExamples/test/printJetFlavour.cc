// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"
#include "DataFormats/Common/interface/Ref.h"

#include "SimDataFormats/JetMatching/interface/JetFlavour.h"

class printJetFlavour : public edm::EDAnalyzer {
  public:
    typedef reco::JetFloatAssociation::Container JetBCEnergyRatioCollection;

    explicit printJetFlavour(const edm::ParameterSet & );
    ~printJetFlavour() {};
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  private:

    typedef std::vector<std::pair<reco::CaloJetRef, reco::JetFlavour> > JetTagVector;

    edm::InputTag sourceByValue_;
    edm::InputTag sourceBratio_;
    edm::InputTag sourceCratio_;
    edm::Handle<JetTagVector>                theTagByValue;
    edm::Handle<JetBCEnergyRatioCollection>  theBratioValue;
    edm::Handle<JetBCEnergyRatioCollection>  theCratioValue;

};

// system include files
#include <memory>
#include <string>
#include <iostream>
#include <vector>

using namespace std;
using namespace reco;
using namespace edm;

printJetFlavour::printJetFlavour(const edm::ParameterSet& iConfig)
{
  sourceByValue_ = iConfig.getParameter<InputTag> ("srcByValue");
  sourceBratio_  = iConfig.getParameter<InputTag> ("srcBratio" );
  sourceCratio_  = iConfig.getParameter<InputTag> ("srcCratio" );
}

void printJetFlavour::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  cout << "[printJetFlavour] analysing event " << iEvent.id() << endl;
  
  try {
    iEvent.getByLabel (sourceByValue_, theTagByValue );
    iEvent.getByLabel (sourceBratio_ , theBratioValue);
    iEvent.getByLabel (sourceCratio_ , theCratioValue);
  } catch(std::exception& ce) {
    cerr << "[printJetFlavour] caught std::exception " << ce.what() << endl;
    return;
  }
  
  cout << "-------------------- Jet Flavour by Value ------------------------" << endl;
  for ( JetTagVector::const_iterator j  = theTagByValue->begin();
                                     j != theTagByValue->end();
                                     j ++ ) {
    const CaloJetRef aJet  = (*j).first;   
    const JetFlavour aFlav = (*j).second;

    printf("[printJetFlavour 1] (pt,eta,phi) jet = %7.3f %6.3f %6.3f | parton = %7.3f %6.3f %6.3f | %2d\n",
             aJet.get()->et(),
             aJet.get()->eta(),
             aJet.get()->phi(), 
             aFlav.getLorentzVector().pt(), 
             aFlav.getLorentzVector().eta(),
             aFlav.getLorentzVector().phi(), 
             aFlav.getFlavour()
          );
  }

  cout << "-------------------- GenJet Bratio ------------------------" << endl;
  for ( JetBCEnergyRatioCollection::const_iterator it  = theBratioValue->begin(); 
                                                   it != theBratioValue->end();
                                                   it ++) {  
   
    const Jet &jet = *(it->first);
    printf("printJetFlavour 2] (pt,eta,phi) jet = %7.3f %6.3f %6.3f | bRatio = %7.5f \n",
             jet.et(),
             jet.eta(),
             jet.phi(),
             it->second
          );
  }
}

DEFINE_FWK_MODULE( printJetFlavour );
