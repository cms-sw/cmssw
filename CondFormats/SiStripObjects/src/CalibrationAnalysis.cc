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
CalibrationAnalysis::CalibrationAnalysis( const uint32_t& key, const bool& deconv, int calchan ) 
  : CommissioningAnalysis(key,"CalibrationAnalysis"),
    amplitude_(2,VFloat(128,sistrip::invalid_)),
    tail_(2,VFloat(128,sistrip::invalid_)),
    riseTime_(2,VFloat(128,sistrip::invalid_)),
    timeConstant_(2,VFloat(128,sistrip::invalid_)),
    smearing_(2,VFloat(128,sistrip::invalid_)),
    chi2_(2,VFloat(128,sistrip::invalid_)),
    mean_amplitude_(2,sistrip::invalid_),
    mean_tail_(2,sistrip::invalid_),
    mean_riseTime_(2,sistrip::invalid_),
    mean_timeConstant_(2,sistrip::invalid_),
    mean_smearing_(2,sistrip::invalid_),
    mean_chi2_(2,sistrip::invalid_),
    min_amplitude_(2,sistrip::invalid_),
    min_tail_(2,sistrip::invalid_),
    min_riseTime_(2,sistrip::invalid_),
    min_timeConstant_(2,sistrip::invalid_),
    min_smearing_(2,sistrip::invalid_),
    min_chi2_(2,sistrip::invalid_),
    max_amplitude_(2,sistrip::invalid_),
    max_tail_(2,sistrip::invalid_),
    max_riseTime_(2,sistrip::invalid_),
    max_timeConstant_(2,sistrip::invalid_),
    max_smearing_(2,sistrip::invalid_),
    max_chi2_(2,sistrip::invalid_),
    spread_amplitude_(2,sistrip::invalid_),
    spread_tail_(2,sistrip::invalid_),
    spread_riseTime_(2,sistrip::invalid_),
    spread_timeConstant_(2,sistrip::invalid_),
    spread_smearing_(2,sistrip::invalid_),
    spread_chi2_(2,sistrip::invalid_),
    deconv_(deconv),
    calchan_(calchan),
    isScan_(false)
{
  deconv_fitter_ = new TF1("deconv_fitter",fdeconv_convoluted,-50,50,5);
  deconv_fitter_->FixParameter(0,0);
  deconv_fitter_->SetParLimits(1,-100,0);
  deconv_fitter_->SetParLimits(2,0,200);
  deconv_fitter_->SetParLimits(3,5,100);
  deconv_fitter_->FixParameter(3,50);
  deconv_fitter_->SetParLimits(4,0,50);
  deconv_fitter_->SetParameters(0.,-10,0.96,50,20);
  peak_fitter_ = new TF1("peak_fitter",fpeak_convoluted,-50,50,5);
  peak_fitter_->FixParameter(0,0);
  peak_fitter_->SetParLimits(1,-100,0);
  peak_fitter_->SetParLimits(2,0,400);
  peak_fitter_->SetParLimits(3,5,100);
  peak_fitter_->FixParameter(3,50);
  peak_fitter_->SetParLimits(4,0,50);
  peak_fitter_->SetParameters(0.,-10,0.96,50,20);
}
// ----------------------------------------------------------------------------
// 
CalibrationAnalysis::CalibrationAnalysis(const bool& deconv, int calchan) 
  : CommissioningAnalysis("CalibrationAnalysis"),
    amplitude_(2,VFloat(128,sistrip::invalid_)),
    tail_(2,VFloat(128,sistrip::invalid_)),
    riseTime_(2,VFloat(128,sistrip::invalid_)),
    timeConstant_(2,VFloat(128,sistrip::invalid_)),
    smearing_(2,VFloat(128,sistrip::invalid_)),
    chi2_(2,VFloat(128,sistrip::invalid_)),
    mean_amplitude_(2,sistrip::invalid_),
    mean_tail_(2,sistrip::invalid_),
    mean_riseTime_(2,sistrip::invalid_),
    mean_timeConstant_(2,sistrip::invalid_),
    mean_smearing_(2,sistrip::invalid_),
    mean_chi2_(2,sistrip::invalid_),
    min_amplitude_(2,sistrip::invalid_),
    min_tail_(2,sistrip::invalid_),
    min_riseTime_(2,sistrip::invalid_),
    min_timeConstant_(2,sistrip::invalid_),
    min_smearing_(2,sistrip::invalid_),
    min_chi2_(2,sistrip::invalid_),
    max_amplitude_(2,sistrip::invalid_),
    max_tail_(2,sistrip::invalid_),
    max_riseTime_(2,sistrip::invalid_),
    max_timeConstant_(2,sistrip::invalid_),
    max_smearing_(2,sistrip::invalid_),
    max_chi2_(2,sistrip::invalid_),
    spread_amplitude_(2,sistrip::invalid_),
    spread_tail_(2,sistrip::invalid_),
    spread_riseTime_(2,sistrip::invalid_),
    spread_timeConstant_(2,sistrip::invalid_),
    spread_smearing_(2,sistrip::invalid_),
    spread_chi2_(2,sistrip::invalid_),
    deconv_(deconv),
    calchan_(calchan),
    isScan_(false)
{
  deconv_fitter_ = new TF1("deconv_fitter",fdeconv_convoluted,-50,50,5);
  deconv_fitter_->FixParameter(0,0);
  deconv_fitter_->SetParLimits(1,-100,0);
  deconv_fitter_->SetParLimits(2,0,200);
  deconv_fitter_->SetParLimits(3,5,100);
  deconv_fitter_->FixParameter(3,50);
  deconv_fitter_->SetParLimits(4,0,50);
  deconv_fitter_->SetParameters(0.,-10,0.96,50,20);
  peak_fitter_ = new TF1("peak_fitter",fpeak_convoluted,-50,50,5);
  peak_fitter_->FixParameter(0,0);
  peak_fitter_->SetParLimits(1,-100,0);
  peak_fitter_->SetParLimits(2,0,400);
  peak_fitter_->SetParLimits(3,5,100);
  peak_fitter_->FixParameter(3,50);
  peak_fitter_->SetParLimits(4,0,50);
  peak_fitter_->SetParameters(0.,-10,0.96,50,20);
}

// ----------------------------------------------------------------------------
// 
void CalibrationAnalysis::print( std::stringstream& ss, uint32_t iapv ) { 
  header( ss );
  ss << " Monitorables for APV number     : " << iapv;
  if ( iapv == 0 ) { ss << " (first of pair)"; }
  else if ( iapv == 1 ) { ss << " (second of pair)"; }
  ss << std::endl;
  ss << " Mean values:" << std::endl;
  ss << " Amplitude of the pulse : " << mean_amplitude_[iapv] << std::endl
     << " Tail amplitude after 150ns : " << mean_tail_[iapv] << std::endl
     << " Rise time : " << mean_riseTime_[iapv] << std::endl
     << " Time constant : " << mean_timeConstant_[iapv] << std::endl
     << " Smearing parameter : " << mean_smearing_[iapv] << std::endl
     << " Chi2 of the fit : " << mean_chi2_[iapv] << std::endl;
}

// ----------------------------------------------------------------------------
// 
void CalibrationAnalysis::reset() {
  amplitude_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  tail_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  riseTime_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  timeConstant_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  smearing_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  chi2_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  mean_amplitude_ = VFloat(2,sistrip::invalid_);
  mean_tail_ = VFloat(2,sistrip::invalid_);
  mean_riseTime_ = VFloat(2,sistrip::invalid_);
  mean_timeConstant_ = VFloat(2,sistrip::invalid_);
  mean_smearing_ = VFloat(2,sistrip::invalid_);
  mean_chi2_ = VFloat(2,sistrip::invalid_);
  min_amplitude_ = VFloat(2,sistrip::invalid_);
  min_tail_ = VFloat(2,sistrip::invalid_);
  min_riseTime_ = VFloat(2,sistrip::invalid_);
  min_timeConstant_ = VFloat(2,sistrip::invalid_);
  min_smearing_ = VFloat(2,sistrip::invalid_);
  min_chi2_ = VFloat(2,sistrip::invalid_);
  max_amplitude_ = VFloat(2,sistrip::invalid_);
  max_tail_ = VFloat(2,sistrip::invalid_);
  max_riseTime_ = VFloat(2,sistrip::invalid_);
  max_timeConstant_ = VFloat(2,sistrip::invalid_);
  max_smearing_ = VFloat(2,sistrip::invalid_);
  max_chi2_ = VFloat(2,sistrip::invalid_);
  spread_amplitude_ = VFloat(2,sistrip::invalid_);
  spread_tail_ = VFloat(2,sistrip::invalid_);
  spread_riseTime_ = VFloat(2,sistrip::invalid_);
  spread_timeConstant_ = VFloat(2,sistrip::invalid_);
  spread_smearing_ = VFloat(2,sistrip::invalid_);
  spread_chi2_ = VFloat(2,sistrip::invalid_);
  deconv_fitter_->FixParameter(0,0);
  deconv_fitter_->SetParLimits(1,-100,0);
  deconv_fitter_->SetParLimits(2,0,200);
  deconv_fitter_->SetParLimits(3,5,100);
  deconv_fitter_->FixParameter(3,50);
  deconv_fitter_->SetParLimits(4,0,50);
  deconv_fitter_->SetParameters(0.,-2.82,0.96,50,20);
  peak_fitter_->FixParameter(0,0);
  peak_fitter_->SetParLimits(1,-100,0);
  peak_fitter_->SetParLimits(2,0,400);
  peak_fitter_->SetParLimits(3,5,100);
  peak_fitter_->FixParameter(3,50);
  peak_fitter_->SetParLimits(4,0,50);
  peak_fitter_->SetParameters(0.,-2.82,0.96,50,20);
}

// ----------------------------------------------------------------------------
// 
void CalibrationAnalysis::extract( const std::vector<TH1*>& histos) {
  
  // Check
  if ( histos.size() != 32 && histos.size() !=2 ) {
    edm::LogWarning(mlCommissioning_) << " Unexpected number of histograms: " << histos.size();
  }
  
  // Extract
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  unsigned int cnt = 0;
  for ( ; ihis != histos.end(); ihis++,cnt++ ) {
    
    // Check pointer
    if ( !(*ihis) ) {
      edm::LogWarning(mlCommissioning_) << " NULL pointer to histogram!";
      continue;
    }
    
    // Check name
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::CALIBRATION && title.runType() != sistrip::CALIBRATION_DECO &&
         title.runType() != sistrip::CALIBRATION_SCAN && title.runType() != sistrip::CALIBRATION_SCAN_DECO ) {
      edm::LogWarning(mlCommissioning_) 
	<< " Unexpected commissioning task: "
	<< SiStripEnumsAndStrings::runType(title.runType());
      continue;
    }
    isScan_ = (title.runType()==sistrip::CALIBRATION_SCAN || title.runType()==sistrip::CALIBRATION_SCAN_DECO);
    
    // Extract calibration histo
    histo_[cnt].first = *ihis;
    histo_[cnt].second = (*ihis)->GetTitle();
    
  }
  
}

// ----------------------------------------------------------------------------
// 
void CalibrationAnalysis::analyse() { 

  float Amean[2]   = {0.,0.};
  float Amin[2]    = {2000.,2000.};
  float Amax[2]    = {0.,0.};
  float Aspread[2] = {0.,0.};
  float Tmean[2]   = {0.,0.};
  float Tmin[2]    = {2000.,2000.};
  float Tmax[2]    = {0.,0.};
  float Tspread[2] = {0.,0.};
  float Rmean[2]   = {0.,0.};
  float Rmin[2]    = {2000.,2000.};
  float Rmax[2]    = {0.,0.};
  float Rspread[2] = {0.,0.};
  float Cmean[2]   = {0.,0.};
  float Cmin[2]    = {2000.,2000.};
  float Cmax[2]    = {0.,0.};
  float Cspread[2] = {0.,0.};
  float Smean[2]   = {0.,0.};
  float Smin[2]    = {2000.,2000.};
  float Smax[2]    = {0.,0.};
  float Sspread[2] = {0.,0.};
  float Kmean[2]   = {0.,0.};
  float Kmin[2]    = {2000000.,2000000.};
  float Kmax[2]    = {0.,0.};
  float Kspread[2] = {0.,0.};
 
  unsigned int upperLimit = isScan_ ? 2 : 32;
  float nStrips = isScan_ ? 1. : 16.;
  for(unsigned int i=0;i<upperLimit;++i) {
    if ( !histo_[i].first ) {
      edm::LogWarning(mlCommissioning_) << " NULL pointer to histogram!" ;
      return;
    }
     
    // determine which APV and strip is being looked at.
    std::string title = histo_[i].first->GetName();
    int apv = 0;
    int strip = 0;
    if(title.find("STRIP")!=std::string::npos && title.find("Apv")!=std::string::npos) {
      strip = atoi(title.c_str()+title.find("STRIP")+6);
      apv = (atoi(title.c_str()+title.find("Apv")+3))%2;
    } else {
      strip = (atoi(title.c_str()+title.find_last_of("_")+1))%16;
      apv = (atoi(title.c_str()+title.find_last_of("_")+1))/16;
      if(title.find("Apv")!=std::string::npos) {
        apv = (atoi(title.c_str()+title.find("Apv")+3))%2;
	strip = strip*8 + calchan_;
      } else {
        edm::LogWarning(mlCommissioning_) << " Malformed histogram title! Strip/APV not retreived: " 
                                          << title;
      }
    }
    
    LogDebug(mlCommissioning_) << "start the calibration analysis for APV " << apv << " strip " << strip;
    // rescale the plot
    correctDistribution(histo_[i].first);
    
    // amplitude
    amplitude_[apv][strip] = histo_[i].first->GetMaximum();
    
    // rise time
    int bin_a=0, bin_b=0, bin_c=0;
    for(int bin = 1; bin<= histo_[i].first->GetNbinsX() && bin_b == 0; ++bin) {
      if(histo_[i].first->GetBinContent(bin)<0.1*amplitude_[apv][strip]) bin_a = bin;
      if(histo_[i].first->GetBinContent(bin)<0.6*amplitude_[apv][strip]) bin_c = bin;
      if(histo_[i].first->GetBinContent(bin)>0.99*amplitude_[apv][strip]) bin_b = bin;
    }
    histo_[i].first->Fit("gaus","0Q","",histo_[i].first->GetBinCenter(bin_b)-25,histo_[i].first->GetBinCenter(bin_b)+25);
    float time_max = ((TF1*)(TVirtualFitter::GetFitter()->GetUserFunc()))->GetMaximumX();
    histo_[i].first->Fit("gaus","0Q","",0,histo_[i].first->GetBinCenter(bin_c));
    TF1* tmp_f = (TF1*)(TVirtualFitter::GetFitter()->GetUserFunc());
    float time_start = tmp_f->GetParameter(1)-3*tmp_f->GetParameter(2);
    riseTime_[apv][strip] = time_max - time_start;
    
    // tail 125 ns after the maximum
    int lastBin = histo_[i].first->FindBin(histo_[i].first->GetBinCenter(histo_[i].first->GetMaximumBin())+125);
    if(lastBin>histo_[i].first->GetNbinsX()-2) lastBin = histo_[i].first->GetNbinsX()-2;
    tail_[apv][strip] = 100*histo_[i].first->GetBinContent(lastBin)/histo_[i].first->GetMaximum();
  
    // perform the fit for the next quantities
    TF1* fit = fitPulse(histo_[i].first);
    
    // time constant
    timeConstant_[apv][strip] = fit->GetParameter(3);
  
    // smearing
    smearing_[apv][strip] = fit->GetParameter(4);
  
    // chi2
    chi2_[apv][strip] = fit->GetChisquare();
    
    LogDebug(mlCommissioning_) << "Results: " << chi2_[apv][strip] << " " << smearing_[apv][strip] << " " 
                               << timeConstant_[apv][strip] << " " << riseTime_[apv][strip] << " " 
  			       << tail_[apv][strip] << " " << amplitude_[apv][strip];
			       
    //compute mean, max, min, spread
    Amean[apv] += amplitude_[apv][strip]/nStrips;
    Amin[apv] = Amin[apv]<amplitude_[apv][strip] ? Amin[apv] : amplitude_[apv][strip];
    Amax[apv] = Amax[apv]>amplitude_[apv][strip] ? Amax[apv] : amplitude_[apv][strip];
    Aspread[apv] += amplitude_[apv][strip]*amplitude_[apv][strip]/nStrips;
    Tmean[apv] += tail_[apv][strip]/nStrips;
    Tmin[apv] = Tmin[apv]<tail_[apv][strip] ? Tmin[apv] : tail_[apv][strip];
    Tmax[apv] = Tmax[apv]>tail_[apv][strip] ? Tmax[apv] : tail_[apv][strip];
    Tspread[apv] += tail_[apv][strip]*tail_[apv][strip]/nStrips;
    Rmean[apv] += riseTime_[apv][strip]/nStrips;
    Rmin[apv] = Rmin[apv]<riseTime_[apv][strip] ? Rmin[apv] : riseTime_[apv][strip];
    Rmax[apv] = Rmax[apv]>riseTime_[apv][strip] ? Rmax[apv] : riseTime_[apv][strip];
    Rspread[apv] += riseTime_[apv][strip]*riseTime_[apv][strip]/nStrips;
    Cmean[apv] += timeConstant_[apv][strip]/nStrips;
    Cmin[apv] = Cmin[apv]<timeConstant_[apv][strip] ? Cmin[apv] : timeConstant_[apv][strip];
    Cmax[apv] = Cmax[apv]>timeConstant_[apv][strip] ? Cmax[apv] : timeConstant_[apv][strip];
    Cspread[apv] += timeConstant_[apv][strip]*timeConstant_[apv][strip]/nStrips;
    Smean[apv] += smearing_[apv][strip]/nStrips;
    Smin[apv] = Smin[apv]<smearing_[apv][strip] ? Smin[apv] : smearing_[apv][strip];
    Smax[apv] = Smax[apv]>smearing_[apv][strip] ? Smax[apv] : smearing_[apv][strip];
    Sspread[apv] += smearing_[apv][strip]*smearing_[apv][strip]/nStrips;
    Kmean[apv] += chi2_[apv][strip]/nStrips;
    Kmin[apv] = Kmin[apv]<chi2_[apv][strip] ? Kmin[apv] : chi2_[apv][strip];
    Kmax[apv] = Kmax[apv]>chi2_[apv][strip] ? Kmax[apv] : chi2_[apv][strip];
    Kspread[apv] += chi2_[apv][strip]*chi2_[apv][strip]/nStrips;
  }
				
  // fill the mean, max, min, spread, ... histograms.
  for(int i=0;i<2;++i) {
    mean_amplitude_[i] = Amean[i];
    mean_tail_[i] = Tmean[i];
    mean_riseTime_[i] = Rmean[i];
    mean_timeConstant_[i] = Cmean[i];
    mean_smearing_[i] = Smean[i];
    mean_chi2_[i] = Kmean[i];
    min_amplitude_[i] = Amin[i];
    min_tail_[i] = Tmin[i];
    min_riseTime_[i] = Rmin[i];
    min_timeConstant_[i] = Cmin[i];
    min_smearing_[i] = Smin[i];
    min_chi2_[i] = Kmin[i];
    max_amplitude_[i] = Amax[i];
    max_tail_[i] = Tmax[i];
    max_riseTime_[i] = Rmax[i];
    max_timeConstant_[i] = Cmax[i];
    max_smearing_[i] = Smax[i];
    max_chi2_[i] = Kmax[i];
    spread_amplitude_[i] = sqrt(fabs(Aspread[i]-Amean[i]*Amean[i]));
    spread_tail_[i] = sqrt(fabs(Tspread[i]-Tmean[i]*Tmean[i]));
    spread_riseTime_[i] = sqrt(fabs(Rspread[i]-Rmean[i]*Rmean[i]));
    spread_timeConstant_[i] = sqrt(fabs(Cspread[i]-Cmean[i]*Cmean[i]));
    spread_smearing_[i] = sqrt(fabs(Sspread[i]-Smean[i]*Smean[i]));
    spread_chi2_[i] = sqrt(fabs(Kspread[i]-Kmean[i]*Kmean[i]));
  }
}

// ----------------------------------------------------------------------------
//
void CalibrationAnalysis::correctDistribution(TH1* histo) const
{
  // return the curve
  histo->Scale(-1);
  if(isScan_) histo->Scale(1/16.);
}

// ----------------------------------------------------------------------------
//
TF1* CalibrationAnalysis::fitPulse(TH1* histo) 
{
  if(!histo) return 0;
  if (deconv_) {
    histo->Fit(deconv_fitter_,"Q");
    return deconv_fitter_; 
  } else {
    histo->Fit(peak_fitter_,"Q");
    return peak_fitter_; 
  } 
}
