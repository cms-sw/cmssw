#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HcalDigi/interface/HcalLaserDigi.h"
#include "RecoLocalCalo/HcalLaserReco/src/HcalLaserUnpacker.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <iostream>
#include <fstream>


class HcalLaserReco : public edm::EDProducer {
public:
  explicit HcalLaserReco(const edm::ParameterSet& ps);
  virtual ~HcalLaserReco();
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
private:
  int qdctdcFed_;
  HcalLaserUnpacker unpacker_;
  edm::InputTag fedRawDataCollectionTag_;
};

HcalLaserReco::HcalLaserReco(edm::ParameterSet const& conf):
  qdctdcFed_(conf.getUntrackedParameter<int>("QADCTDCFED",8)),
  fedRawDataCollectionTag_(conf.getParameter<edm::InputTag>("fedRawDataCollectionTag"))
{
  
    produces<HcalLaserDigi>();
}

// Virtual destructor needed.
HcalLaserReco::~HcalLaserReco() { }  

// Functions that gets called by framework every event
void HcalLaserReco::produce(edm::Event& e, const edm::EventSetup&)
{
  // Step A: Get Inputs 
  edm::Handle<FEDRawDataCollection> rawraw;
  e.getByLabel(fedRawDataCollectionTag_, rawraw);

  // Step B: Create empty output    
  std::auto_ptr<HcalLaserDigi>
    digi(new HcalLaserDigi);
    
  if (qdctdcFed_ >=0) {
    // Step C: unpack all requested FEDs
    const FEDRawData& fed = rawraw->FEDData(qdctdcFed_);
    unpacker_.unpack(fed,*digi);
  }
  
  // Step D: Put outputs into event
  e.put(digi);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HcalLaserReco);

