///
/// \class l1t::L1TGlobalInputTester.cc
///
/// Description: Dump/Analyze Input Collections for GT.
///
/// Implementation:
///    Based off of Michael Mulhearn's YellowParamTester
///
/// \author: Brian Winer Ohio State
///


//
//  This simple module simply retreives the YellowParams object from the event
//  setup, and sends its payload as an INFO message, for debugging purposes.
//


#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
//#include "FWCore/ParameterSet/interface/InputTag.h"

// system include files
#include <iomanip>

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"


#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

using namespace edm;
using namespace std;

namespace l1t {

  // class declaration
  class L1TGlobalInputTester : public edm::EDAnalyzer {
  public:
    explicit L1TGlobalInputTester(const edm::ParameterSet&);
    virtual ~L1TGlobalInputTester(){};
    virtual void analyze(const edm::Event&, const edm::EventSetup&);  
    
    EDGetToken egToken;
    EDGetToken muToken;
    EDGetToken tauToken;
    EDGetToken jetToken;
    EDGetToken etsumToken;
  };

  L1TGlobalInputTester::L1TGlobalInputTester(const edm::ParameterSet& iConfig)
  {
      egToken    = consumes<BXVector<l1t::EGamma>>(iConfig.getParameter<InputTag>("egInputTag"));
      muToken    = consumes<BXVector<l1t::EGamma>>(iConfig.getParameter<InputTag>("muInputTag"));
      tauToken   = consumes<BXVector<l1t::EGamma>>(iConfig.getParameter<InputTag>("tauInputTag"));
      jetToken   = consumes<BXVector<l1t::EGamma>>(iConfig.getParameter<InputTag>("jetInputTag"));
      etsumToken = consumes<BXVector<l1t::EGamma>>(iConfig.getParameter<InputTag>("etsumInputTag"));
  }
  
  // loop over events
  void L1TGlobalInputTester::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup){

    
 //inputs
  Handle<BXVector<l1t::EGamma>> egammas;
  iEvent.getByToken(egToken,egammas);

  Handle<BXVector<l1t::Muon>> muons;
  iEvent.getByToken(egToken,egammas);
 
   Handle<BXVector<l1t::Tau>> taus;
   iEvent.getByToken(tauToken,taus);

  Handle<BXVector<l1t::Jet>> jets;
  iEvent.getByToken(jetToken,jets);
 
  Handle<BXVector<l1t::EtSum>> etsums;
  iEvent.getByToken(etsumToken,etsums); 
 
 //Loop over BX
    for(int i = egammas->getFirstBX(); i <= egammas->getLastBX(); ++i) {
    
       //Loop over EGamma
       for(std::vector<l1t::EGamma>::const_iterator eg = egammas->begin(i); eg != egammas->end(i); ++eg) {
           printf("BX=%i  Pt %i Eta %i Phi %i Qual %i  \n",i,eg->hwPt(),eg->hwEta(),eg->hwPhi(),eg->hwQual());
       }    

       //Loop over Muons
       for(std::vector<l1t::Muon>::const_iterator mu = muons->begin(i); mu != muons->end(i); ++mu) {
          printf("BX=%i  Pt %i Eta %i Phi %i Qual %i  \n",i,mu->hwPt(),mu->hwEta(),mu->hwPhi(),mu->hwQual());
       }
                  //Dump Content
       for(std::vector<l1t::Tau>::const_iterator tau = taus->begin(i); tau != taus->end(i); ++tau) {
           printf("BX=%i  Pt %i Eta %i Phi %i Qual %i  \n",i,tau->hwPt(),tau->hwEta(),tau->hwPhi(),tau->hwQual());
       }        

       //Loop over Jets
       for(std::vector<l1t::Jet>::const_iterator jet = jets->begin(i); jet != jets->end(i); ++jet) {
          printf("BX=%i  Pt %i Eta %i Phi %i Qual %i \n",i,jet->hwPt(),jet->hwEta(),jet->hwPhi(),jet->hwQual());
       }
                  //Dump Content
       for(std::vector<l1t::EtSum>::const_iterator etsum = etsums->begin(i); etsum != etsums->end(i); ++etsum) {
           printf("BX=%i  Pt %i Eta %i Phi %i Qual %i \n",i,etsum->hwPt(),etsum->hwEta(),etsum->hwPhi(),etsum->hwQual());
       }        


    }
  }

}


DEFINE_FWK_MODULE(l1t::L1TGlobalInputTester);

