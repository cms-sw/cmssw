#include "CondFormats/SiStripObjects/interface/CalibrationAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripPulseShape.h"
#include "TProfile.h"
#include "TF1.h"
#include "TH1.h"
#include "TVirtualFitter.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
CalibrationAnalysis::CalibrationAnalysis( const uint32_t& key, const bool& deconv ) 
  : CommissioningAnalysis(key,"CalibrationAnalysis"),
    amplitude_(sistrip::invalid_),
    tail_(sistrip::invalid_),
    riseTime_(sistrip::invalid_),
    timeConstant_(sistrip::invalid_),
    smearing_(sistrip::invalid_),
    chi2_(sistrip::invalid_),
    histo_(0,""),
    deconv_(deconv)
{
  deconv_fitter_ = new TF1("deconv_fitter",fdeconv_convoluted,-50,50,5);
  deconv_fitter_->FixParameter(0,0);
  deconv_fitter_->SetParLimits(1,-10,10);
  deconv_fitter_->SetParLimits(2,0,200);
  deconv_fitter_->SetParLimits(3,5,100);
  deconv_fitter_->FixParameter(3,50);
  deconv_fitter_->SetParLimits(4,0,50);
  deconv_fitter_->SetParameters(0.,-2.82,0.96,50,20);
  peak_fitter_ = new TF1("peak_fitter",fpeak_convoluted,-50,50,5);
  peak_fitter_->FixParameter(0,0);
  peak_fitter_->SetParLimits(1,-10,10);
  peak_fitter_->SetParLimits(2,0,200);
  peak_fitter_->SetParLimits(3,5,100);
  peak_fitter_->FixParameter(3,50);
  peak_fitter_->SetParLimits(4,0,50);
  peak_fitter_->SetParameters(0.,-2.82,0.96,50,20);
}
// ----------------------------------------------------------------------------
// 
CalibrationAnalysis::CalibrationAnalysis(const bool& deconv) 
  : CommissioningAnalysis("CalibrationAnalysis"),
    amplitude_(sistrip::invalid_),
    tail_(sistrip::invalid_),
    riseTime_(sistrip::invalid_),
    timeConstant_(sistrip::invalid_),
    smearing_(sistrip::invalid_),
    chi2_(sistrip::invalid_),
    histo_(0,""),
    deconv_(deconv)
{
  deconv_fitter_ = new TF1("deconv_fitter",fdeconv_convoluted,-50,50,5);
  deconv_fitter_->FixParameter(0,0);
  deconv_fitter_->SetParLimits(1,-10,10);
  deconv_fitter_->SetParLimits(2,0,200);
  deconv_fitter_->SetParLimits(3,5,100);
  deconv_fitter_->FixParameter(3,50);
  deconv_fitter_->SetParLimits(4,0,50);
  deconv_fitter_->SetParameters(0.,-2.82,0.96,50,20);
  peak_fitter_ = new TF1("peak_fitter",fpeak_convoluted,-50,50,5);
  peak_fitter_->FixParameter(0,0);
  peak_fitter_->SetParLimits(1,-10,10);
  peak_fitter_->SetParLimits(2,0,200);
  peak_fitter_->SetParLimits(3,5,100);
  peak_fitter_->FixParameter(3,50);
  peak_fitter_->SetParLimits(4,0,50);
  peak_fitter_->SetParameters(0.,-2.82,0.96,50,20);
}

// ----------------------------------------------------------------------------
// 
void CalibrationAnalysis::print( std::stringstream& ss, uint32_t not_used ) { 
  header( ss );
  ss << " Amplitude of the pulse : " << amplitude_ << std::endl
     << " Tail amplitude after 150ns : " << tail_ << std::endl
     << " Rise time : " << riseTime_ << std::endl
     << " Time constant : " << timeConstant_ << std::endl
     << " Smearing parameter : " << smearing_ << std::endl
     << " Chi2 of the fit : " << chi2_ << std::endl;
}

// ----------------------------------------------------------------------------
// 
void CalibrationAnalysis::reset() {
  amplitude_ = sistrip::invalid_;
  tail_ = sistrip::invalid_;
  riseTime_ = sistrip::invalid_;
  timeConstant_ = sistrip::invalid_;
  smearing_ = sistrip::invalid_;
  chi2_ = sistrip::invalid_;
  histo_ = Histo(0,"");
  deconv_fitter_->FixParameter(0,0);
  deconv_fitter_->SetParLimits(1,-10,10);
  deconv_fitter_->SetParLimits(2,0,200);
  deconv_fitter_->SetParLimits(3,5,100);
  deconv_fitter_->FixParameter(3,50);
  deconv_fitter_->SetParLimits(4,0,50);
  deconv_fitter_->SetParameters(0.,-2.82,0.96,50,20);
  peak_fitter_->FixParameter(0,0);
  peak_fitter_->SetParLimits(1,-10,10);
  peak_fitter_->SetParLimits(2,0,200);
  peak_fitter_->SetParLimits(3,5,100);
  peak_fitter_->FixParameter(3,50);
  peak_fitter_->SetParLimits(4,0,50);
  peak_fitter_->SetParameters(0.,-2.82,0.96,50,20);
}

// ----------------------------------------------------------------------------
// 
void CalibrationAnalysis::extract( const std::vector<TH1*>& histos) {
  
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
    
    // Extract calibration histo
    histo_.first = *ihis;
    histo_.second = (*ihis)->GetName();
    
  }
  
}

// ----------------------------------------------------------------------------
// 
void CalibrationAnalysis::analyse() { 
  if ( !histo_.first ) {
    edm::LogWarning(mlCommissioning_) << " NULL pointer to histogram!" ;
    return;
  }
  
  // rescale the plot
  correctDistribution(histo_.first);
  
  // amplitude
  amplitude_ = histo_.first->GetMaximum();
  
  // tail 
  int lastBin = deconv_ ? 80 : 65;
  tail_ = 100*histo_.first->GetBinContent(lastBin)/histo_.first->GetMaximum();
  
  // rise time
  int bin_a=0, bin_b=0, bin_c=0;
  for(int bin = 1; bin<= histo_.first->GetNbinsX() && bin_b == 0; ++bin) {
      if(histo_.first->GetBinContent(bin)<0.1*amplitude_) bin_a = bin;
      if(histo_.first->GetBinContent(bin)<0.6*amplitude_) bin_c = bin;
      if(histo_.first->GetBinContent(bin)>0.99*amplitude_) bin_b = bin;
    }
  histo_.first->Fit("gaus","0Q","",histo_.first->GetBinCenter(bin_b)-25,histo_.first->GetBinCenter(bin_b)+25);
  float time_max = ((TF1*)(TVirtualFitter::GetFitter()->GetUserFunc()))->GetMaximumX();
  histo_.first->Fit("gaus","0Q","",0,histo_.first->GetBinCenter(bin_c));
  TF1* tmp_f = (TF1*)(TVirtualFitter::GetFitter()->GetUserFunc());
  float time_start = tmp_f->GetParameter(1)-3*tmp_f->GetParameter(2);
  riseTime_ = time_max - time_start;

  // perform the fit for the next quantities
  TF1* fit = fitPulse(histo_.first);
  
  // time constant
  timeConstant_ = fit->GetParameter(3);

  // smearing
  smearing_ = fit->GetParameter(4);

  // chi2
  chi2_ = fit->GetChisquare();
  
}

// ----------------------------------------------------------------------------
//
void CalibrationAnalysis::correctDistribution(TH1* histo) const
{
  histo->Scale(-1);
  histo->GetXaxis()->SetLimits(histo->GetXaxis()->GetXmin()*3.125,
                               histo->GetXaxis()->GetXmax()*3.125);		   
}

// ----------------------------------------------------------------------------
//
TF1* CalibrationAnalysis::fitPulse(TH1* histo) 
{
  if(!histo) return 0;
  if (deconv_) {
    histo->Fit(deconv_fitter_,"0Q");
    return deconv_fitter_; 
  } else {
    histo->Fit(peak_fitter_,"0Q");
    return peak_fitter_; 
  } 
}
