#include "EventFilter/ESRawToDigi/interface/ESRawToDigiTB.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "TBDataFormats/ESTBRawData/interface/ESDCCHeaderBlock.h"
#include "TBDataFormats/ESTBRawData/interface/ESKCHIPBlock.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h" 

#include <iostream>

ESRawToDigiTB::ESRawToDigiTB(ParameterSet const& ps) 
{

  label_ = ps.getParameter<string>("Label");
  instanceName_ = ps.getParameter<string>("InstanceES");
  ESdigiCollection_ = ps.getParameter<string>("ESdigiCollection");
  debug_ = ps.getUntrackedParameter<bool>("debugMode", false);

  ESUnpackerTB_ = new ESUnpackerTB(ps);

  produces<ESRawDataCollection>();
  produces<ESLocalRawDataCollection>();
  produces<ESDigiCollection>();
}

ESRawToDigiTB::~ESRawToDigiTB(){

  delete ESUnpackerTB_;

}

void ESRawToDigiTB::produce(edm::Event& e, const edm::EventSetup& es) {

  pair<int,int> ESFEDIds = FEDNumbering::getPreShowerFEDIds();

  // Input
  Handle<FEDRawDataCollection> rawdata;
  e.getByLabel(label_, instanceName_, rawdata);
  if (!rawdata.isValid()) {
    LogDebug("") << "ESRawToDigiTB : Error! can't get rawdata!" << std::endl;
  }
  
  // Output
  auto_ptr<ESRawDataCollection> productDCC(new ESRawDataCollection);
  auto_ptr<ESLocalRawDataCollection> productKCHIP(new ESLocalRawDataCollection);
  auto_ptr<ESDigiCollection> productDigis(new ESDigiCollection);    

  for (int fedId=ESFEDIds.first; fedId<=ESFEDIds.second; ++fedId) {
    const FEDRawData& fedRawData = rawdata->FEDData(fedId);
    ESUnpackerTB_->interpretRawData(fedId, fedRawData, *productDCC, *productKCHIP, *productDigis);
    if (debug_) cout<<"FED : "<<fedId<<" Data size : "<<fedRawData.size()<<" (Bytes)"<<endl;
  }   
   
  e.put(productDCC);
  e.put(productKCHIP);
  e.put(productDigis, ESdigiCollection_);
}
