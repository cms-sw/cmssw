#include "DQM/SiStripCommissioningAnalysis/interface/FineDelayAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripPulseShape.h"
#include "TProfile.h"
#include "TF1.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
FineDelayAnalysis::FineDelayAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,"FineDelayAnalysis"),
    max_(sistrip::invalid_),
    error_(sistrip::invalid_),
    histo_(0,"")
{
  deconv_fitter_ = new TF1("deconv_fitter",fdeconv_convoluted,-50,50,5);
  deconv_fitter_->FixParameter(0,0);
  deconv_fitter_->SetParLimits(1,-10,10);
  deconv_fitter_->SetParLimits(2,0,200);
  deconv_fitter_->SetParLimits(3,5,100);
  deconv_fitter_->FixParameter(3,50);
  deconv_fitter_->SetParLimits(4,0,50);
  deconv_fitter_->SetParameters(0.,-2.82,0.96,50,20);
}
// ----------------------------------------------------------------------------
// 
FineDelayAnalysis::FineDelayAnalysis() 
  : CommissioningAnalysis("FineDelayAnalysis"),
    max_(sistrip::invalid_),
    error_(sistrip::invalid_),
    histo_(0,"")
{
  deconv_fitter_ = new TF1("deconv_fitter",fdeconv_convoluted,-50,50,5);
  deconv_fitter_->FixParameter(0,0);
  deconv_fitter_->SetParLimits(1,-10,10);
  deconv_fitter_->SetParLimits(2,0,200);
  deconv_fitter_->SetParLimits(3,5,100);
  deconv_fitter_->FixParameter(3,50);
  deconv_fitter_->SetParLimits(4,0,50);
  deconv_fitter_->SetParameters(0.,-2.82,0.96,50,20);
}

// ----------------------------------------------------------------------------
// 
void FineDelayAnalysis::print( std::stringstream& ss, uint32_t not_used ) { 
  header( ss );
  ss << " Delay corresponding to the maximum of the pulse : " << max_ << std::endl
     << " Error on the position (from the fit)            : " << error_ << std::endl;
}

// ----------------------------------------------------------------------------
// 
void FineDelayAnalysis::reset() {
  error_ = sistrip::invalid_;
  max_ = sistrip::invalid_;
  histo_ = Histo(0,"");
  deconv_fitter_->FixParameter(0,0);
  deconv_fitter_->SetParLimits(1,-10,10);
  deconv_fitter_->SetParLimits(2,0,200);
  deconv_fitter_->SetParLimits(3,5,100);
  deconv_fitter_->FixParameter(3,50);
  deconv_fitter_->SetParLimits(4,0,50);
  deconv_fitter_->SetParameters(0.,-2.82,0.96,50,20);
}

// ----------------------------------------------------------------------------
// 
void FineDelayAnalysis::extract( const std::vector<TH1*>& histos) {
  
  // Check
  if ( histos.size() != 1 ) {
    edm::LogWarning(mlCommissioning_) << " Unexpected number of histograms: " << histos.size();
  }
  
  // Extract
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check pointer
    if ( !(*ihis) ) {
      edm::LogWarning(mlCommissioning_) << " NULL pointer to histogram!";
      continue;
    }
    
    // Check name
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::FINE_DELAY ) {
      edm::LogWarning(mlCommissioning_) 
	<< " Unexpected commissioning task: "
	<< SiStripEnumsAndStrings::runType(title.runType());
      continue;
    }
    
    // Extract timing histo
    histo_.first = *ihis;
    histo_.second = (*ihis)->GetName();
    
  }
  
}

// ----------------------------------------------------------------------------
// 
void FineDelayAnalysis::analyse() { 
  if ( !histo_.first ) {
    edm::LogWarning(mlCommissioning_) << " NULL pointer to histogram!" ;
    return;
  }
  
  histo_.first->Fit(deconv_fitter_,"QL");
  
  // Set monitorables
  max_ = deconv_fitter_->GetMaximumX();
  error_ = deconv_fitter_->GetParError(1);

}
