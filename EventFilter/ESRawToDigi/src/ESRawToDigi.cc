#include "EventFilter/ESRawToDigi/interface/ESRawToDigi.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h" 
#include "DataFormats/EcalRawData/interface/ESDCCHeaderBlock.h"
#include "DataFormats/EcalRawData/interface/ESKCHIPBlock.h"

#include <iostream>

ESRawToDigi::ESRawToDigi(ParameterSet const& ps) 
{
  sourceTag_ = ps.getParameter<edm::InputTag>("sourceTag");
  ESdigiCollection_ = ps.getParameter<string>("ESdigiCollection");
  debug_ = ps.getUntrackedParameter<bool>("debugMode", false);

  ESUnpacker_ = new ESUnpacker(ps);

  produces<ESRawDataCollection>();
  produces<ESLocalRawDataCollection>();
  produces<ESDigiCollection>();
}

ESRawToDigi::~ESRawToDigi(){

  delete ESUnpacker_;

}

void ESRawToDigi::produce(edm::Event& e, const edm::EventSetup& es) {
  // Input
  Handle<FEDRawDataCollection> rawdata;
  e.getByLabel(sourceTag_, rawdata);
  if (!rawdata.isValid()) {
    LogDebug("") << "ESRawToDigi : Error! can't get rawdata!" << std::endl;
  }
  
  // Output
  auto_ptr<ESRawDataCollection> productDCC(new ESRawDataCollection);
  auto_ptr<ESLocalRawDataCollection> productKCHIP(new ESLocalRawDataCollection);
  auto_ptr<ESDigiCollection> productDigis(new ESDigiCollection);  
  
  ESDigiCollection digis;

  for (int fedId=FEDNumbering::MINPreShowerFEDID; fedId<=FEDNumbering::MAXPreShowerFEDID; ++fedId) {

    const FEDRawData& fedRawData = rawdata->FEDData(fedId);
    ESUnpacker_->interpretRawData(fedId, fedRawData, *productDCC, *productKCHIP, *productDigis);

    if (debug_) cout<<"FED : "<<fedId<<" Data size : "<<fedRawData.size()<<" (Bytes)"<<endl;
  }   
   
  e.put(productDCC);
  e.put(productKCHIP);
  e.put(productDigis, ESdigiCollection_);
}
