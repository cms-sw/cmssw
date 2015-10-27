#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include <iostream>

namespace edm {
  class EventSetup;
}

//
// class declaration
//
class BunchSpacingProducer : public edm::stream::EDProducer<> {

public:

  explicit BunchSpacingProducer(const edm::ParameterSet&);

  ~BunchSpacingProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&) override final;

  static void fillDescriptions( edm::ConfigurationDescriptions & ) ;
  
private:
  
  edm::EDGetTokenT<int> bunchSpacing_;
  unsigned int bunchSpacingOverride_;
  bool overRide_;
};

//
// constructors and destructor
//


BunchSpacingProducer::
BunchSpacingProducer::BunchSpacingProducer(const edm::ParameterSet& iConfig)
{
  // register your products
  produces<unsigned int>();
  bunchSpacing_ = consumes<int>(edm::InputTag("addPileupInfo","bunchSpacing"));
  overRide_=false;
  if ( iConfig.exists("overrideBunchSpacing") ) {
    overRide_= iConfig.getParameter<bool>("overrideBunchSpacing");
    if ( overRide_) {
      bunchSpacingOverride_=iConfig.getParameter<unsigned int>("bunchSpacingOverride");
    }
  }
}

BunchSpacingProducer::~BunchSpacingProducer(){ 
}

//
// member functions
//
void BunchSpacingProducer::produce(edm::Event& e, const edm::EventSetup& iSetup)
{ 
  if ( overRide_ ) {
    std::auto_ptr<unsigned int> pOut1(new unsigned int);
    *pOut1=bunchSpacingOverride_;
    e.put(pOut1);
    return;
  }

  unsigned int bunchSpacing=50;
  unsigned int run=e.run();

  if ( e.isRealData()) {
    if ( ( run > 252126 && run != 254833 )|| 
	run == 178003 ||
	run == 178004 ||
	run == 209089 ||
	run == 209106 ||
	run == 209109 ||
	run == 209146 ||
	run == 209148 ||
	run == 209151) {
      bunchSpacing = 25;
    }
  }
  else{
    edm::Handle<int> bunchSpacingH;
    e.getByToken(bunchSpacing_,bunchSpacingH);
    bunchSpacing = *bunchSpacingH;
  }

  std::auto_ptr<unsigned int> pOut1(new unsigned int);
  *pOut1=bunchSpacing;
  e.put(pOut1);
  return;
}

void BunchSpacingProducer::fillDescriptions( edm::ConfigurationDescriptions & descriptions )
{
  edm::ParameterSetDescription desc ;
  desc.add<bool>("overrideBunchSpacing",false); // true for prompt reco
  desc.add<unsigned int>("bunchSpacingOverride",25); // override value
  
  descriptions.add("bunchSpacingProducer",desc) ;
}



#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BunchSpacingProducer);
