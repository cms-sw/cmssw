// File: SiStripZeroSuppressionAlgorithm.cc
// Description:  An algorithm for CMS track reconstruction.
// Author:  Domenico Giordano

#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripZeroSuppressionAlgorithm.h"

using namespace std;

SiStripZeroSuppressionAlgorithm::SiStripZeroSuppressionAlgorithm(const edm::ParameterSet& conf) : 
  conf_(conf),  
  SiStripPedestalsService_(conf),
  ZeroSuppressionMode_(conf.getParameter<string>("ZeroSuppressionMode")),
  CMNSubtractionMode_(conf.getParameter<string>("CommonModeNoiseSubtractionMode")){
    
  //------------------------
  if ( ZeroSuppressionMode_ == "SiStripFedZeroSuppression" ) {
    SiStripZeroSuppressor_ = new SiStripZeroSuppressor(); 
    validZeroSuppression_ = true;
  } 
  else {
    std::cout << "[SiStripZeroSuppressionAlgorithm] No valid strip ZeroSuppressor selected, possible ZeroSuppressor: FEDSiStripZeroSuppression" << endl;
    validZeroSuppression_ = false;
  }

  //------------------------
  if ( CMNSubtractionMode_ != "CMNSubraction_Median") { 
    std::cout << "[SiStripZeroSuppressionAlgorithm] No valid strip ZeroSuppressor selected, possible ZeroSuppressor: FEDSiStripZeroSuppression" << endl;
    validCMNSubtraction_ = false;
  } 
  else {
    SiStripCommonModeNoiseSubtractor_ = new SiStripCommonModeNoiseSubtractor(CMNSubtractionMode_);
  }
  //------------------------

  SiStripPedestalsSubtractor_ = new SiStripPedestalsSubtractor();
}

SiStripZeroSuppressionAlgorithm::~SiStripZeroSuppressionAlgorithm() {
  if ( SiStripZeroSuppressor_ != 0 ) 
    delete SiStripZeroSuppressor_;
  if ( SiStripCommonModeNoiseSubtractor_ != 0 ) 
    delete SiStripCommonModeNoiseSubtractor_;
  if ( SiStripPedestalsSubtractor_ != 0 ) 
    delete SiStripPedestalsSubtractor_;
}

void SiStripZeroSuppressionAlgorithm::configure( const edm::EventSetup& es ) {
    SiStripPedestalsService_.configure(es);
    SiStripPedestalsSubtractor_->setSiStripPedestalsService(SiStripPedestalsService_);
    SiStripZeroSuppressor_->setSiStripPedestalsService(SiStripPedestalsService_);
} 

void SiStripZeroSuppressionAlgorithm::run(std::string RawDigiType, 
					  const edm::DetSetVector<SiStripRawDigi>& input,
					  edm::DetSetVector<SiStripDigi>& output){

  
  if ( validZeroSuppression_ && validCMNSubtraction_) {
    int number_detunits        = 0;
    int number_localstripdigis = 0;

    //loop on all detset inside the input collection
    edm::DetSetVector<SiStripRawDigi>::const_iterator DSViter=input.begin();
    for (; DSViter!=input.end();DSViter++){
      ++number_detunits;
      std::cout << "[SiStripZeroSuppressionAlgorithm::run] DetID " << DSViter->id << std::endl;
    
      //Create or pass the new DetSet<SiStripDigi>
      edm::DetSet<SiStripDigi>& ssd = output.find_or_insert(DSViter->id);
      
      //Create a temporary edm::DetSet<SiStripRawDigi> 
      std::vector<int16_t> vssRd((*DSViter).data.size());

      if ( RawDigiType == "VirginRaw" ) {
	SiStripPedestalsSubtractor_->subtract(*DSViter,vssRd);
	SiStripCommonModeNoiseSubtractor_->subtract(vssRd);
	SiStripZeroSuppressor_->suppress(vssRd,ssd);
      } 
      else if ( RawDigiType == "ProcessedRaw" ){
	SiStripZeroSuppressor_->suppress((*DSViter),ssd);	
      }
      else{
	//FIXME
	//insert throw exception
      }
      number_localstripdigis += ssd.data.size();     
    }
    cout << "[SiStripZeroSuppressionAlgorithm] execution in mode " << ZeroSuppressionMode_ << " generating " << number_localstripdigis << " StripDigi in " << number_detunits << " DetUnits." << endl; 
  }
  
};
