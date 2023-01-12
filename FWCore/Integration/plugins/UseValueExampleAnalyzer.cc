// -*- C++ -*-
//
// Package:    Integration
// Class:      UseValueExampleAnalyzer
//
/**\class UseValueExampleAnalyzer UseValueExampleAnalyzer.cc FWCore/Integration/test/UseValueExampleAnalyzer.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Chris D Jones
//         Created:  Thu Sep  8 03:55:42 EDT 2005
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "ValueExample.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class decleration
//

class UseValueExampleAnalyzer : public edm::global::EDAnalyzer<> {
public:
  explicit UseValueExampleAnalyzer(const edm::ParameterSet&);

  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const final;

private:
  // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
UseValueExampleAnalyzer::UseValueExampleAnalyzer(const edm::ParameterSet& /* iConfig */) {
  //now do what ever initialization is needed
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void UseValueExampleAnalyzer::analyze(edm::StreamID,
                                      const edm::Event& /* iEvent */,
                                      const edm::EventSetup& /* iSetup*/) const {
  std::cout << " value from service " << edm::Service<ValueExample>()->value() << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(UseValueExampleAnalyzer);
