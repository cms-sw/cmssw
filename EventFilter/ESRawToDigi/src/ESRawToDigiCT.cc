#include "EventFilter/ESRawToDigi/interface/ESRawToDigiCT.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "TBDataFormats/ESTBRawData/interface/ESDCCHeaderBlock.h"
#include "TBDataFormats/ESTBRawData/interface/ESKCHIPBlock.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h" 

#include <iostream>

ESRawToDigiCT::ESRawToDigiCT(ParameterSet const& ps) :
  fedUnpackList_(ps.getUntrackedParameter<std::vector<int> >("FEDs", vector<int>()))
{

  if (fedUnpackList_.empty())
    for (int i=FEDNumbering::getEcalFEDIds().first; i<=FEDNumbering::getEcalFEDIds().second; i++)
      fedUnpackList_.push_back(i);

  for (unsigned int i=0; i<fedUnpackList_.size(); i++)
    cout<<"ESRawToDigi will unpack FED : "<<fedUnpackList_[i]<<endl;

  label_ = ps.getParameter<string>("Label");
  instanceName_ = ps.getParameter<string>("InstanceES");
  ESdigiCollection_ = ps.getParameter<string>("ESdigiCollection");
  debug_ = ps.getUntrackedParameter<bool>("debugMode", false);

  ESUnpackerCT_ = new ESUnpackerCT(ps);

  produces<ESRawDataCollection>();
  produces<ESLocalRawDataCollection>();
  produces<ESDigiCollection>();
}

ESRawToDigiCT::~ESRawToDigiCT(){

  delete ESUnpackerCT_;

}

void ESRawToDigiCT::produce(edm::Event& e, const edm::EventSetup& es) {

  // Input
  Handle<FEDRawDataCollection> rawdata;
  e.getByLabel(label_, instanceName_, rawdata);
  if (!rawdata.isValid()) {
    LogDebug("") << "ESRawToDigiCT : Error! can't get rawdata!" << std::endl;
  }
  
  // Output
  auto_ptr<ESRawDataCollection> productDCC(new ESRawDataCollection);
  auto_ptr<ESLocalRawDataCollection> productKCHIP(new ESLocalRawDataCollection);
  auto_ptr<ESDigiCollection> productDigis(new ESDigiCollection);    

  for (vector<int>::const_iterator i=fedUnpackList_.begin(); i!=fedUnpackList_.end(); ++i) {
    const FEDRawData& fedRawData = rawdata->FEDData(*i);
    if (debug_) cout<<"FED : "<<*i<<" Data size : "<<fedRawData.size()<<" (Bytes)"<<endl;
    if (fedRawData.size() != 0) 
      ESUnpackerCT_->interpretRawData(*i, fedRawData, *productDCC, *productKCHIP, *productDigis);
  }   
   
  e.put(productDCC);
  e.put(productKCHIP);
  e.put(productDigis, ESdigiCollection_);
}
