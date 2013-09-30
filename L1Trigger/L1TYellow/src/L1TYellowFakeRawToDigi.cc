// -*- C++ -*-
//
// Package:    L1TYellowFakeDigiProducer
// Class:      L1TYellowFakeDigiProducer
// 
/**\class L1TYellowFakeRawToDigi

 Description: Emulation of Fictitious Level-1 Yellow Trigger for demonstration purposes

 Implementation:
     Produces L1TYellowInput
*/
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TYellow/interface/L1TYellowDigi.h"

using namespace std;
//using namespace l1t;

//
// class declaration
//

class L1TYellowFakeRawToDigi : public edm::EDProducer {
public:
  explicit L1TYellowFakeRawToDigi(const edm::ParameterSet&);
  ~L1TYellowFakeRawToDigi();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  virtual void beginRun(edm::Run&, edm::EventSetup const&);
  virtual void endRun(edm::Run&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);  
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
L1TYellowFakeRawToDigi::L1TYellowFakeRawToDigi(const edm::ParameterSet& iConfig)
{
  //register your products
  produces<L1TYellowDigiCollection>();
}


L1TYellowFakeRawToDigi::~L1TYellowFakeRawToDigi()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1TYellowFakeRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  cout << "L1TYellowFakeRawToDigi::produce function called...\n";

  std::auto_ptr<L1TYellowDigiCollection> outColl (new L1TYellowDigiCollection);
  L1TYellowDigi iout;

  iout.setRawData(20);
  outColl->push_back(iout);

  iEvent.put(outColl); 
}

// ------------ method called once each job just before starting event loop  ------------
void 
L1TYellowFakeRawToDigi::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TYellowFakeRawToDigi::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void 
L1TYellowFakeRawToDigi::beginRun(edm::Run&, edm::EventSetup const&)
{

}

// ------------ method called when ending the processing of a run  ------------
void 
L1TYellowFakeRawToDigi::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
L1TYellowFakeRawToDigi::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
L1TYellowFakeRawToDigi::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TYellowFakeRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TYellowFakeRawToDigi);
