#include "DQM/SiStripCommissioningAnalysis/interface/CalibrationAlgorithm.h"
#include "CondFormats/SiStripObjects/interface/CalibrationAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripCommissioningAnalysis/interface/SiStripPulseShape.h"
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
CalibrationAlgorithm::CalibrationAlgorithm( const edm::ParameterSet & pset, CalibrationAnalysis* const anal ) 
  : CommissioningAlgorithm(anal),
    deconv_fitter_(0),
    peak_fitter_(0),
    cal_(0)
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
void CalibrationAlgorithm::extract( const std::vector<TH1*>& histos) {
  if ( !anal() ) {
    edm::LogWarning(mlCommissioning_)
      << "[CalibrationAlgorithm::" << __func__ << "]"
      << " NULL pointer to base Analysis object!";
    return; 
  }

  CommissioningAnalysis* tmp = const_cast<CommissioningAnalysis*>( anal() );
  cal_ = dynamic_cast<CalibrationAnalysis*>( tmp );
  if ( !cal_ ) {
    edm::LogWarning(mlCommissioning_)
      << "[CalibrationAlgorithm::" << __func__ << "]"
      << " NULL pointer to derived Analysis object!";
    return; 
  }

  // Check number of histograms
  if ( histos.size() != 32 && histos.size() !=2 ) {
    cal_->addErrorCode(sistrip::numberOfHistos_);
  }
  
  // Extract FED key from histo title
  if ( !histos.empty() ) { cal_->fedKey( extractFedKey( histos.front() ) ); }
  
  // Extract histograms
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  unsigned int cnt = 0;
  for ( ; ihis != histos.end(); ihis++,cnt++ ) {
    
    // Check for NULL pointer
    if ( !(*ihis) ) { continue; }
    
    // Check name
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::CALIBRATION && 
	 title.runType() != sistrip::CALIBRATION_DECO &&
         title.runType() != sistrip::CALIBRATION_SCAN && 
	 title.runType() != sistrip::CALIBRATION_SCAN_DECO ) {
      cal_->addErrorCode(sistrip::unexpectedTask_);
      continue;
    }
    
    cal_->isScan_ = ( title.runType() == sistrip::CALIBRATION_SCAN || 
		      title.runType() == sistrip::CALIBRATION_SCAN_DECO ); 
    
    // Extract calibration histo
    histo_[cnt].first = *ihis;
    histo_[cnt].second = (*ihis)->GetTitle();
  }
  
}

// ----------------------------------------------------------------------------
// 
void CalibrationAlgorithm::analyse() { 

  if ( !cal_ ) {
    edm::LogWarning(mlCommissioning_)
      << "[CalibrationAlgorithm::" << __func__ << "]"
      << " NULL pointer to derived Analysis object!";
    return; 
  }

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
 
  unsigned int upperLimit = cal_->isScan_ ? 2 : 32;
  float nStrips = cal_->isScan_ ? 1. : 16.;
  for(unsigned int i=0;i<upperLimit;++i) {
    if ( !histo_[i].first ) {
      edm::LogWarning(mlCommissioning_) 
	<< " NULL pointer to histogram " << i << "!";
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
	strip = strip*8 + cal_->calchan_;
      } else {
        edm::LogWarning(mlCommissioning_) 
	  << " Malformed histogram title! Strip/APV not retreived: " 
	  << title;
      }
    }
    
    // rescale the plot
    correctDistribution(histo_[i].first);
    
    // amplitude
    cal_->amplitude_[apv][strip] = histo_[i].first->GetMaximum();
    
    // rise time
    cal_->riseTime_[apv][strip] = maximum(histo_[i].first) - turnOn(histo_[i].first);
    
    // tail 125 ns after the maximum
    int lastBin = histo_[i].first->FindBin(histo_[i].first->GetBinCenter(histo_[i].first->GetMaximumBin())+125);
    if(lastBin>histo_[i].first->GetNbinsX()-4) lastBin = histo_[i].first->GetNbinsX()-4;
     if(histo_[i].first->GetMaximum()!=0)
       cal_->tail_[apv][strip] = 100*histo_[i].first->GetBinContent(lastBin)/histo_[i].first->GetMaximum();
     else
       cal_->tail_[apv][strip] = 100;
  
    // perform the fit for the next quantities
    TF1* fit = fitPulse(histo_[i].first);
    
    // time constant
    cal_->timeConstant_[apv][strip] = fit->GetParameter(3);
  
    // smearing
    cal_->smearing_[apv][strip] = fit->GetParameter(4);
  
    // chi2
    cal_->chi2_[apv][strip] = fit->GetChisquare();
    
    //compute mean, max, min, spread
    Amean[apv] += cal_->amplitude_[apv][strip]/nStrips;
    Amin[apv] = Amin[apv]<cal_->amplitude_[apv][strip] ? Amin[apv] : cal_->amplitude_[apv][strip];
    Amax[apv] = Amax[apv]>cal_->amplitude_[apv][strip] ? Amax[apv] : cal_->amplitude_[apv][strip];
    Aspread[apv] += cal_->amplitude_[apv][strip]*cal_->amplitude_[apv][strip]/nStrips;
    Tmean[apv] += cal_->tail_[apv][strip]/nStrips;
    Tmin[apv] = Tmin[apv]<cal_->tail_[apv][strip] ? Tmin[apv] : cal_->tail_[apv][strip];
    Tmax[apv] = Tmax[apv]>cal_->tail_[apv][strip] ? Tmax[apv] : cal_->tail_[apv][strip];
    Tspread[apv] += cal_->tail_[apv][strip]*cal_->tail_[apv][strip]/nStrips;
    Rmean[apv] += cal_->riseTime_[apv][strip]/nStrips;
    Rmin[apv] = Rmin[apv]<cal_->riseTime_[apv][strip] ? Rmin[apv] : cal_->riseTime_[apv][strip];
    Rmax[apv] = Rmax[apv]>cal_->riseTime_[apv][strip] ? Rmax[apv] : cal_->riseTime_[apv][strip];
    Rspread[apv] += cal_->riseTime_[apv][strip]*cal_->riseTime_[apv][strip]/nStrips;
    Cmean[apv] += cal_->timeConstant_[apv][strip]/nStrips;
    Cmin[apv] = Cmin[apv]<cal_->timeConstant_[apv][strip] ? Cmin[apv] : cal_->timeConstant_[apv][strip];
    Cmax[apv] = Cmax[apv]>cal_->timeConstant_[apv][strip] ? Cmax[apv] : cal_->timeConstant_[apv][strip];
    Cspread[apv] += cal_->timeConstant_[apv][strip]*cal_->timeConstant_[apv][strip]/nStrips;
    Smean[apv] += cal_->smearing_[apv][strip]/nStrips;
    Smin[apv] = Smin[apv]<cal_->smearing_[apv][strip] ? Smin[apv] : cal_->smearing_[apv][strip];
    Smax[apv] = Smax[apv]>cal_->smearing_[apv][strip] ? Smax[apv] : cal_->smearing_[apv][strip];
    Sspread[apv] += cal_->smearing_[apv][strip]*cal_->smearing_[apv][strip]/nStrips;
    Kmean[apv] += cal_->chi2_[apv][strip]/nStrips;
    Kmin[apv] = Kmin[apv]<cal_->chi2_[apv][strip] ? Kmin[apv] : cal_->chi2_[apv][strip];
    Kmax[apv] = Kmax[apv]>cal_->chi2_[apv][strip] ? Kmax[apv] : cal_->chi2_[apv][strip];
    Kspread[apv] += cal_->chi2_[apv][strip]*cal_->chi2_[apv][strip]/nStrips;
  }
				
  // fill the mean, max, min, spread, ... histograms.
  for(int i=0;i<2;++i) {
    cal_->mean_amplitude_[i] = Amean[i];
    cal_->mean_tail_[i] = Tmean[i];
    cal_->mean_riseTime_[i] = Rmean[i];
    cal_->mean_timeConstant_[i] = Cmean[i];
    cal_->mean_smearing_[i] = Smean[i];
    cal_->mean_chi2_[i] = Kmean[i];
    cal_->min_amplitude_[i] = Amin[i];
    cal_->min_tail_[i] = Tmin[i];
    cal_->min_riseTime_[i] = Rmin[i];
    cal_->min_timeConstant_[i] = Cmin[i];
    cal_->min_smearing_[i] = Smin[i];
    cal_->min_chi2_[i] = Kmin[i];
    cal_->max_amplitude_[i] = Amax[i];
    cal_->max_tail_[i] = Tmax[i];
    cal_->max_riseTime_[i] = Rmax[i];
    cal_->max_timeConstant_[i] = Cmax[i];
    cal_->max_smearing_[i] = Smax[i];
    cal_->max_chi2_[i] = Kmax[i];
    cal_->spread_amplitude_[i] = sqrt(fabs(Aspread[i]-Amean[i]*Amean[i]));
    cal_->spread_tail_[i] = sqrt(fabs(Tspread[i]-Tmean[i]*Tmean[i]));
    cal_->spread_riseTime_[i] = sqrt(fabs(Rspread[i]-Rmean[i]*Rmean[i]));
    cal_->spread_timeConstant_[i] = sqrt(fabs(Cspread[i]-Cmean[i]*Cmean[i]));
    cal_->spread_smearing_[i] = sqrt(fabs(Sspread[i]-Smean[i]*Smean[i]));
    cal_->spread_chi2_[i] = sqrt(fabs(Kspread[i]-Kmean[i]*Kmean[i]));
  }
}

// ----------------------------------------------------------------------------
//
void CalibrationAlgorithm::correctDistribution( TH1* histo ) const
{
  // return the curve
  histo->Scale(-1);
  if ( cal_ ) { if( cal_->isScan_ ) histo->Scale(1/16.); }
}

// ----------------------------------------------------------------------------
//
TF1* CalibrationAlgorithm::fitPulse( TH1* histo, 
				     float rangeLow, 
				     float rangeHigh ) 
{
  if(!cal_) return 0;
  if(!histo) return 0;
  float noise = 4.;
  float N = round(histo->GetMaximum()/125.);
  float error = sqrt(2*N)*noise;
  //float error = sqrt(2)*noise*N; // ?????????
  for(int i=1;i<=histo->GetNbinsX();++i) {
    histo->SetBinError(i,error);
  }
  if (cal_->deconv_) {
    if(rangeLow>rangeHigh)
      histo->Fit(deconv_fitter_,"Q");
    else 
      histo->Fit(deconv_fitter_,"0Q","",rangeLow,rangeHigh);
    return deconv_fitter_; 
  } else {
    if(rangeLow>rangeHigh)
      histo->Fit(peak_fitter_,"Q");
    else 
      histo->Fit(peak_fitter_,"0Q","",rangeLow,rangeHigh);
    return peak_fitter_; 
  } 
}

// ----------------------------------------------------------------------------
//
float CalibrationAlgorithm::maximum( TH1* h ) {
    int bin = h->GetMaximumBin();
    // fit around the maximum with the detector response and take the max from the fit
    TF1* fit = fitPulse(h,h->GetBinCenter(bin)-25,h->GetBinCenter(bin)+25);
    return fit->GetMaximumX();
}

// ----------------------------------------------------------------------------
//
float CalibrationAlgorithm::turnOn( TH1* h ) {
    // localize the rising edge
    int bin=1;
    float amplitude = h->GetMaximum();
    for(; bin<= h->GetNbinsX() && h->GetBinContent(bin)<0.4*amplitude; ++bin) {}
    float end = h->GetBinLowEdge(bin);
    // fit the rising edge with a sigmoid
    TF1* sigmoid = new TF1("sigmoid","[0]/(1+exp(-[1]*(x-[2])))+[3]",0,end);
    sigmoid->SetParLimits(0,amplitude/10.,amplitude);
    sigmoid->SetParLimits(1,0.05,0.5);
    sigmoid->SetParLimits(2,end-10,end+10);
    sigmoid->SetParLimits(3,-amplitude/10.,amplitude/10.);
    sigmoid->SetParameters(amplitude/2.,0.1,end,0.);
    h->Fit(sigmoid,"0QR");
    // return the point where the fit = 3% signal.
    float time = 0.;
    float base = sigmoid->GetMinimum(0,end);
    for(;time<end && (sigmoid->Eval(time)-base)<0.03*(amplitude-base);time += 0.1) {}
    delete sigmoid;
    return time-0.05;
}

