// -*- C++ -*-
//
// Package:    L1Trigger/L1TYellow
// Class:      L1TYellowConfigESProducer
// 
/**\class L1TYellowConfigESProducer L1TYellowConfigESProducer.cc L1Trigger/L1TYellow/src/L1TYellowConfigESProducer.cc

 Description:  This is part of the fictitious Yellow trigger emulation for demonstration purposes.

 This uses the parameters set in a config file to fill the ConfData/L1TYellow/L1TYellowConfig object.  
 Other modules can retreive this object in same manner whether it was filled from config file or 
 the conditions database.

 Implementation:
     [Notes on implementation]
*/
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/L1TYellow/interface/L1TYellowParams.h"

using namespace std;
//using namespace l1t;

//
// class declaration
//

class L1TYellowConfigESProducer : public edm::ESProducer {
public:
  explicit L1TYellowConfigESProducer(const edm::ParameterSet&);
  ~L1TYellowConfigESProducer();
  
private:
  std::auto_ptr<L1TYellowParams> produce(const FooRecord&);

  
};


//using namespace l1t;

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
L1TYellowConfigESProducer::L1TYellowConfigESProducer(const edm::ParameterSet& iConfig)
{
  //register your products
  produces<L1TYellowDigiCollection>();
}


L1TYellowConfigESProducer::~L1TYellowConfigESProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1TYellowConfigESProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  cout << "L1TYellowConfigESProducer::produce function called...\n";

  std::auto_ptr<L1TYellowDigiCollection> outColl (new L1TYellowDigiCollection);
  L1TYellowDigi iout;

  iout.setRawData(20);
  outColl->push_back(iout);

  iEvent.put(outColl); 
}

// ------------ method called once each job just before starting event loop  ------------
void 
L1TYellowConfigESProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TYellowConfigESProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void 
L1TYellowConfigESProducer::beginRun(edm::Run&, edm::EventSetup const&)
{

}

// ------------ method called when ending the processing of a run  ------------
void 
L1TYellowConfigESProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
L1TYellowConfigESProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
L1TYellowConfigESProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TYellowConfigESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TYellowConfigESProducer);
