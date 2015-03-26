#include "EventFilter/ESRawToDigi/interface/ESRawToDigi.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h" 
#include "DataFormats/EcalRawData/interface/ESDCCHeaderBlock.h"
#include "DataFormats/EcalRawData/interface/ESKCHIPBlock.h"

#include <iostream>

ESRawToDigi::ESRawToDigi(edm::ParameterSet const& ps) 
{
   
  edm::InputTag sourceTag = ps.getParameter<edm::InputTag>("sourceTag");
  ESdigiCollection_ = ps.getParameter<std::string>("ESdigiCollection");
  regional_         = ps.getUntrackedParameter<bool>("DoRegional",false);
  edm::InputTag fedsListLabel     
      = ps.getUntrackedParameter<edm::InputTag>("ESFedsListLabel", edm::InputTag(":esfedslist"));
  debug_            = ps.getUntrackedParameter<bool>("debugMode", false);

  ESUnpacker_ = new ESUnpacker(ps);

  produces<ESRawDataCollection>();
  produces<ESLocalRawDataCollection>();
  produces<ESDigiCollection>();
  dataToken_=consumes<FEDRawDataCollection>(sourceTag);
  if (regional_){
      fedsToken_=consumes<ESListOfFEDS>(fedsListLabel);
  }
}

ESRawToDigi::~ESRawToDigi(){

  delete ESUnpacker_;

}

void ESRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("sourceTag",edm::InputTag("rawDataCollector"));
  desc.addUntracked<bool>("debugMode",false);
  desc.add<std::string>("InstanceES","");
  desc.add<edm::FileInPath>("LookupTable",edm::FileInPath("EventFilter/ESDigiToRaw/data/ES_lookup_table.dat"));
  desc.add<std::string>("ESdigiCollection","");
  descriptions.add("esRawToDigi",desc);
}

void ESRawToDigi::produce(edm::Event& e, const edm::EventSetup& es) {

  // Input
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByToken(dataToken_, rawdata);
  if (!rawdata.isValid()) {
    LogDebug("") << "ESRawToDigi : Error! can't get rawdata!" << std::endl;
  }
  
  std::vector<int> esFeds_to_unpack;
  if (regional_) {
    edm::Handle<ESListOfFEDS> fedslist;
    e.getByToken(fedsToken_, fedslist);
    esFeds_to_unpack = fedslist->GetList();
  }

  // Output
  std::auto_ptr<ESRawDataCollection> productDCC(new ESRawDataCollection);
  std::auto_ptr<ESLocalRawDataCollection> productKCHIP(new ESLocalRawDataCollection);
  std::auto_ptr<ESDigiCollection> productDigis(new ESDigiCollection);  
  
  ESDigiCollection digis;

  if (regional_) {
    for (unsigned int i=0; i<esFeds_to_unpack.size(); ++i) {
      
      const FEDRawData& fedRawData = rawdata->FEDData(esFeds_to_unpack[i]);
      ESUnpacker_->interpretRawData(esFeds_to_unpack[i], fedRawData, *productDCC, *productKCHIP, *productDigis);
      
      if (debug_) std::cout<<"FED : "<<esFeds_to_unpack[i]<<" Data size : "<<fedRawData.size()<<" (Bytes)"<<std::endl;
    }   
  } else {
    for (int fedId=FEDNumbering::MINPreShowerFEDID; fedId<=FEDNumbering::MAXPreShowerFEDID; ++fedId) {
      
      const FEDRawData& fedRawData = rawdata->FEDData(fedId);
      ESUnpacker_->interpretRawData(fedId, fedRawData, *productDCC, *productKCHIP, *productDigis);
      
      if (debug_) std::cout<<"FED : "<<fedId<<" Data size : "<<fedRawData.size()<<" (Bytes)"<<std::endl;
    }   
  }

  e.put(productDCC);
  e.put(productKCHIP);
  e.put(productDigis, ESdigiCollection_);
}

