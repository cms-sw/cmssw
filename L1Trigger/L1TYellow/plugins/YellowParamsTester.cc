///
/// \class l1t::YellowParamsTestser
///
/// Description: Tester for the YellowParams written to the event setup.
///
/// Implementation:
///    Demonstrates how to implement a ConfFormats class tester.
///
/// \author: Michael Mulhearn - UC Davis
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

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TYellow/interface/YellowParams.h"
#include "CondFormats/DataRecord/interface/L1TYellowParamsRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

using namespace edm;
using namespace std;

namespace l1t {

  // class declaration
  class YellowParamsTester : public edm::EDAnalyzer {
  public:
    explicit YellowParamsTester(const edm::ParameterSet&){};
    virtual ~YellowParamsTester(){};
    virtual void analyze(const edm::Event&, const edm::EventSetup&);  
  };
  
  // loop over events
  void YellowParamsTester::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup){

    
    edm::ESHandle< YellowParams > yparams ;
    evSetup.get< L1TYellowParamsRcd >().get( yparams ) ;
    
    LogInfo("l1t|yellow") << " Dumping Yellow Parameters content:\n" << (*yparams) << std::endl;
    
  }

}


DEFINE_FWK_MODULE(l1t::YellowParamsTester);

