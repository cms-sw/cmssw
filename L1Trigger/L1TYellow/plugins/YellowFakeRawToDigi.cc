///
/// \class l1t::YellowFakeRawToDigi
///
/// Description: Fake Raw-to-Digi producer for the fictitious Yellow trigger.
///
/// Implementation:
///    Produces fake Digis for use by the Yellow trigger.
///
/// \author: Michael Mulhearn - UC Davis
///


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TYellow/interface/YellowDigi.h"

using namespace std;

//
// class declaration
//

namespace l1t {

  class YellowFakeRawToDigi : public edm::EDProducer {
  public:
    explicit YellowFakeRawToDigi(const edm::ParameterSet&);
    ~YellowFakeRawToDigi();
    
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

}

using namespace l1t;

YellowFakeRawToDigi::YellowFakeRawToDigi(const edm::ParameterSet& iConfig){
  //register your products
  produces<YellowDigiCollection>();
}


YellowFakeRawToDigi::~YellowFakeRawToDigi(){
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
YellowFakeRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::auto_ptr<YellowDigiCollection> outColl (new YellowDigiCollection);
  YellowDigi iout;

  iout.setEt(20);
  iout.setYvar(2.0);
  outColl->push_back(iout);

  iout.setEt(30);
  iout.setYvar(1.0);
  outColl->push_back(iout);

  iout.setEt(25);
  iout.setYvar(1.5);
  outColl->push_back(iout);

  iEvent.put(outColl); 
}

// ------------ method called once each job just before starting event loop  ------------
void 
YellowFakeRawToDigi::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
YellowFakeRawToDigi::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void 
YellowFakeRawToDigi::beginRun(edm::Run&, edm::EventSetup const&)
{

}

// ------------ method called when ending the processing of a run  ------------
void 
YellowFakeRawToDigi::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
YellowFakeRawToDigi::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
YellowFakeRawToDigi::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
YellowFakeRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::YellowFakeRawToDigi);
