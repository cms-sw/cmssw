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
  };

  L1TGlobalInputTester::L1TGlobalInputTester(const edm::ParameterSet& iConfig)
  {
      egToken = consumes<BXVector<l1t::EGamma>>(iConfig.getParameter<InputTag>("egInputTag"));
  }
  
  // loop over events
  void L1TGlobalInputTester::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup){

    
 //inputs
  Handle<BXVector<l1t::EGamma>> egammas;
  iEvent.getByToken(egToken,egammas);
 
 //Loop over BX
    for(int i = egammas->getFirstBX(); i <= egammas->getLastBX(); ++i) {
    
       //Dump Content
       for(std::vector<l1t::EGamma>::const_iterator eg = egammas->begin(i); eg != egammas->end(i); ++eg) {
           printf("BX=%i  Pt %i \n",i,eg->hwPt());
       }    
    }
  }

}


DEFINE_FWK_MODULE(l1t::L1TGlobalInputTester);

