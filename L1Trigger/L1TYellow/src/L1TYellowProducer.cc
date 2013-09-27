// -*- C++ -*-
//
// Package:    L1TYellowProducer
// Class:      L1TYellowProducer
// 
/**\class L1TYellowProducer L1TYellowProducer.cc L1Trigger/L1TYellow/src/L1TYellowProducer.cc

 Description: Emulation of Fictitious Level-1 Yellow Trigger for demonstration purposes

 Implementation:
     [Notes on implementation]
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

#include "DataFormats/L1TYellow/interface/L1TYellowOutput.h"
#include "L1Trigger/L1TYellow/interface/L1TYellowAlg.h"
#include "L1Trigger/L1TYellow/interface/L1TYellowDBParams.h"

using namespace std;
//using namespace l1t;

//
// class declaration
//

class L1TYellowProducer : public edm::EDProducer {
public:
  explicit L1TYellowProducer(const edm::ParameterSet&);
  ~L1TYellowProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  virtual void beginRun(edm::Run&, edm::EventSetup const&);
  virtual void endRun(edm::Run&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
  
  // ----------member data ---------------------------
  l1t::L1TYellowDBParams * dbpars; // Database parameters for the trigger, to be udpated each run
  l1t::L1TYellowAlg * alg; // Algorithm to run per event, depends on database parameters, updated each run.

};


using namespace l1t;

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
L1TYellowProducer::L1TYellowProducer(const edm::ParameterSet& iConfig)
{
  //register your products
  produces<L1TYellowOutputCollection>();
  dbpars = NULL;
  alg = NULL;
}


L1TYellowProducer::~L1TYellowProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1TYellowProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  cout << "L1TYellowProducer::produce function called...\n";

  std::auto_ptr<L1TYellowOutputCollection> outColl (new L1TYellowOutputCollection);
  L1TYellowOutput iout;
  
  if (alg) {
    alg->processEvent(iEvent, *outColl);
  } else {
    // already complained in beginRun, doing nothing now will send empty collection to event, as desired.
  }

  //iout.setRawData(20);  outColl->push_back(iout);
  //iout.setRawData(18);  outColl->push_back(iout);
  //iout.setRawData(30);  outColl->push_back(iout);

  iEvent.put(outColl);
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
L1TYellowProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TYellowProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void 
L1TYellowProducer::beginRun(edm::Run&, edm::EventSetup const&)
{
  // TODO:  retreive DB pars from database, just set here:
  if (dbpars) delete dbpars;
  dbpars = new L1TYellowDBParams;
  dbpars->version = 1;


  // Set the current algorithm version based on DB pars from database:
  if (alg) delete alg;
  alg = NewL1TYellowAlg(*dbpars);

  if (! alg) {
    // we complain here once per run
  }

}

// ------------ method called when ending the processing of a run  ------------
void 
L1TYellowProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
L1TYellowProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
L1TYellowProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TYellowProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TYellowProducer);
