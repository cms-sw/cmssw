#include "EventFilter/ESRawToDigi/interface/ESRawToDigi.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalRawData/interface/EcalListOfFEDS.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h" 
#include "DataFormats/EcalRawData/interface/ESDCCHeaderBlock.h"
#include "DataFormats/EcalRawData/interface/ESKCHIPBlock.h"

#include <iostream>

ESRawToDigi::ESRawToDigi(ParameterSet const& ps) 
{
  sourceTag_        = ps.getParameter<edm::InputTag>("sourceTag");
  ESdigiCollection_ = ps.getParameter<string>("ESdigiCollection");
  regional_         = ps.getUntrackedParameter<bool>("DoRegional",false);
  fedsListLabel_    = ps.getUntrackedParameter<edm::InputTag>("ESFedsListLabel", edm::InputTag(":esfedslist"));
  debug_            = ps.getUntrackedParameter<bool>("debugMode", false);

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
  
  std::vector<int> esFeds_to_unpack;
  if (regional_) {
    edm::Handle<EcalListOfFEDS> fedslist;
    e.getByLabel(fedsListLabel_, fedslist);
    esFeds_to_unpack = fedslist->GetList();
  }

  // Output
  auto_ptr<ESRawDataCollection> productDCC(new ESRawDataCollection);
  auto_ptr<ESLocalRawDataCollection> productKCHIP(new ESLocalRawDataCollection);
  auto_ptr<ESDigiCollection> productDigis(new ESDigiCollection);  
  
  ESDigiCollection digis;

  if (regional_) {
    for (uint i=0; i<esFeds_to_unpack.size(); ++i) {
      
      const FEDRawData& fedRawData = rawdata->FEDData(esFeds_to_unpack[i]);
      ESUnpacker_->interpretRawData(esFeds_to_unpack[i], fedRawData, *productDCC, *productKCHIP, *productDigis);
      
      if (debug_) cout<<"FED : "<<esFeds_to_unpack[i]<<" Data size : "<<fedRawData.size()<<" (Bytes)"<<endl;
    }   
  } else {
    for (int fedId=FEDNumbering::MINPreShowerFEDID; fedId<=FEDNumbering::MAXPreShowerFEDID; ++fedId) {
      
      const FEDRawData& fedRawData = rawdata->FEDData(fedId);
      ESUnpacker_->interpretRawData(fedId, fedRawData, *productDCC, *productKCHIP, *productDigis);
      
      if (debug_) cout<<"FED : "<<fedId<<" Data size : "<<fedRawData.size()<<" (Bytes)"<<endl;
    }   
  }

  e.put(productDCC);
  e.put(productKCHIP);
  e.put(productDigis, ESdigiCollection_);
}
