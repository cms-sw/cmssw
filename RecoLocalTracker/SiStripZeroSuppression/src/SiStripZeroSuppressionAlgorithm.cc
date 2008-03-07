// File: SiStripZeroSuppressionAlgorithm.cc
// Description:  An algorithm for CMS track reconstruction.
// Author:  Domenico Giordano

#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripZeroSuppressionAlgorithm.h"

#include "CommonTools/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripFedZeroSuppression.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripMedianCommonModeNoiseSubtraction.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripFastLinearCommonModeNoiseSubtraction.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripTT6CommonModeNoiseSubtraction.h"

#include "sstream"

SiStripZeroSuppressionAlgorithm::SiStripZeroSuppressionAlgorithm(const edm::ParameterSet& conf) : 
  conf_(conf),  
  ZeroSuppressionMode_(conf.getParameter<std::string>("ZeroSuppressionMode")),
  CMNSubtractionMode_(conf.getParameter<std::string>("CommonModeNoiseSubtractionMode")){
    
  edm::LogInfo("SiStripZeroSuppression") 
    << "[SiStripZeroSuppressionAlgorithm::SiStripZeroSuppressionAlgorithm] Constructing object..."
    << " ZeroSuppressionMode: " << ZeroSuppressionMode_
    << "[SiStripZeroSuppressionAlgorithm::SiStripZeroSuppressionAlgorithm] CMNSubtractionMode: " << CMNSubtractionMode_;

  //------------------------
  if ( ZeroSuppressionMode_ == "SiStripFedZeroSuppression" ) {
    SiStripZeroSuppressor_ = new SiStripFedZeroSuppression(conf.getParameter<uint32_t>("FEDalgorithm")); 
    validZeroSuppression_ = true;
  } 
  else {
    edm::LogError("SiStripZeroSuppression") << "[SiStripZeroSuppressionAlgorithm::SiStripZeroSuppressionAlgorithm] No valid strip ZeroSuppressor selected, possible ZeroSuppressor: SiStripFedZeroSuppression" << std::endl;
    validZeroSuppression_ = false;
  }

  //------------------------
  if ( CMNSubtractionMode_ == "Median") { 
    SiStripCommonModeNoiseSubtractor_ = new SiStripMedianCommonModeNoiseSubtraction();
    validCMNSubtraction_ = true;
  }
  else if ( CMNSubtractionMode_ == "TT6") { 
    SiStripCommonModeNoiseSubtractor_ = new SiStripTT6CommonModeNoiseSubtraction(conf.getParameter<double>("CutToAvoidSignal"));
    validCMNSubtraction_ = true;
  }
  else if ( CMNSubtractionMode_ == "FastLinear") { 
    SiStripCommonModeNoiseSubtractor_ = new SiStripFastLinearCommonModeNoiseSubtraction();
    validCMNSubtraction_ = true;
  }
  else {
    edm::LogError("SiStripZeroSuppression") << "[SiStripZeroSuppressionAlgorithm::SiStripZeroSuppressionAlgorithm] No valid CommonModeNoiseSubtraction Mode selected, possible CMNSubtractionMode: Median or TT6" << std::endl;
    validCMNSubtraction_ = false;
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

void SiStripZeroSuppressionAlgorithm::run(std::string RawDigiType, 
					  const edm::DetSetVector<SiStripRawDigi>& input,
					  std::vector< edm::DetSet<SiStripDigi> >& output,
					  const edm::EventSetup& es){
  

  
  if ( validZeroSuppression_ && validCMNSubtraction_) {
    int number_detunits        = 0;
    int number_localstripdigis = 0;

    //loop on all detset inside the input collection
    edm::DetSetVector<SiStripRawDigi>::const_iterator DSViter=input.begin();
    for (; DSViter!=input.end();DSViter++){
      ++number_detunits;
     
      //Create a new DetSet<SiStripDigi>
      edm::DetSet<SiStripDigi> ssd(DSViter->id);
      
      //Create a temporary edm::DetSet<SiStripRawDigi> 
      std::vector<int16_t> vssRd((*DSViter).data.size());

      if ( RawDigiType == "VirginRaw" ) {
	SiStripPedestalsSubtractor_->subtract(*DSViter,vssRd,es);
	SiStripCommonModeNoiseSubtractor_->subtract(DSViter->id,vssRd,es);
	SiStripZeroSuppressor_->suppress(vssRd,ssd,es);
      } 
      else if ( RawDigiType == "ProcessedRaw" ){
	SiStripZeroSuppressor_->suppress((*DSViter),ssd,es);	
      }
      else{
	//FIXME
	//insert throw exception
      }    
      number_localstripdigis += ssd.data.size();         
      
      if (ssd.data.size())
	output.push_back(ssd);  // insert the DetSet<SiStripDigi> in the  DetSetVec<SiStripDigi> only if there is at least a digi
    }

  }
}

