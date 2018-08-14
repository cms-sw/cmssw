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
#include "TFitResultPtr.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
CalibrationAlgorithm::CalibrationAlgorithm( const edm::ParameterSet & pset, CalibrationAnalysis* const anal ) 
  : CommissioningAlgorithm(anal),
    deconv_fitter_(nullptr),
    peak_fitter_(nullptr),
    cal_(nullptr)
{

  deconv_fitter_ = new TF1("deconv_fitter",fdeconv,0,200,6);
  deconv_fitter_->SetParLimits(0,1,70); //x
  deconv_fitter_->SetParLimits(1,0,40); //z = tau
  deconv_fitter_->SetParLimits(4,15,55); // turn-on-time t_0
  deconv_fitter_->SetParameters(2,25,0.8000,50,40);

  peak_fitter_ = new TF1("peak_fitter",fpeak,0,200,5);
  peak_fitter_->SetParLimits(0,1,70); //x
  peak_fitter_->SetParLimits(1,20,70);//z = tau
  peak_fitter_->SetParLimits(4,15,35); //turn-on-time t_0
  peak_fitter_->SetParameters(17,50,0,5000,20);
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
  // turnOn
  float Omean[2]   = {0.,0.};
  float Omin[2]    = {2000.,2000.};
  float Omax[2]    = {0.,0.};
  float Ospread[2] = {0.,0.};
  // maximum
  float Mmean[2]   = {0.,0.};
  float Mmin[2]    = {2000.,2000.};
  float Mmax[2]    = {0.,0.};
  float Mspread[2] = {0.,0.};
  // undershoot
  float Umean[2]   = {0.,0.};
  float Umin[2]    = {2000.,2000.};
  float Umax[2]    = {0.,0.};
  float Uspread[2] = {0.,0.};
  // baseline
  float Bmean[2]   = {0.,0.};
  float Bmin[2]    = {2000.,2000.};
  float Bmax[2]    = {0.,0.};
  float Bspread[2] = {0.,0.};
 
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

    /// make the fit
    TF1* fit = fitPulse(histo_[i].first);
    float maximum_ampl = fit->GetMaximum();
    float baseline     = fit->Eval(10);
    cal_->amplitude_[apv][strip] = maximum_ampl - baseline;
 
    // rise time
    float peak_time = fit->GetMaximumX();
    float turn_on_time = turnOn(fit);
    float rise_time = peak_time - turn_on_time;
    cal_->riseTime_[apv][strip] = rise_time;

    //turn-on
    cal_->turnOn_[apv][strip] = turn_on_time;

    //maximum
    cal_->maximum_[apv][strip] = peak_time;

    //undershoot
    if (cal_->deconv_)
      cal_->undershoot_[apv][strip] = 100*(fit->GetMinimum()-baseline)/(maximum_ampl - baseline);
    else
      cal_->undershoot_[apv][strip] = 0;

    //baseline
    cal_->baseline_[apv][strip] = 100 * baseline/(maximum_ampl - baseline);
    

    // tail 125 ns after the maximum
    int lastBin = histo_[i].first->FindBin(peak_time + 125);
    if(lastBin>histo_[i].first->GetNbinsX()-4) lastBin = histo_[i].first->GetNbinsX()-4;
    if(histo_[i].first->GetMaximum()!=0)
      cal_->tail_[apv][strip] = 100*(histo_[i].first->GetBinContent(lastBin)-baseline) / (maximum_ampl - baseline);
    else
      cal_->tail_[apv][strip] = 100;
    
    cal_->timeConstant_[apv][strip] = fit->GetParameter(1);
    cal_->smearing_[apv][strip] = 0;
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
    
    Omean[apv] += cal_->turnOn_[apv][strip]/nStrips;
    Omin[apv] = Omin[apv]<cal_->turnOn_[apv][strip] ? Omin[apv] : cal_->turnOn_[apv][strip];
    Omax[apv] = Omax[apv]>cal_->turnOn_[apv][strip] ? Omax[apv] : cal_->turnOn_[apv][strip];
    Ospread[apv] += cal_->turnOn_[apv][strip]*cal_->turnOn_[apv][strip]/nStrips;

    Mmean[apv] += cal_->maximum_[apv][strip]/nStrips;
    Mmin[apv] = Mmin[apv]<cal_->maximum_[apv][strip] ? Mmin[apv] : cal_->maximum_[apv][strip];
    Mmax[apv] = Mmax[apv]>cal_->maximum_[apv][strip] ? Mmax[apv] : cal_->maximum_[apv][strip];
    Mspread[apv] += cal_->maximum_[apv][strip]*cal_->maximum_[apv][strip]/nStrips;

    Umean[apv] += cal_->undershoot_[apv][strip]/nStrips;
    Umin[apv] = Umin[apv]<cal_->undershoot_[apv][strip] ? Umin[apv] : cal_->undershoot_[apv][strip];
    Umax[apv] = Umax[apv]>cal_->undershoot_[apv][strip] ? Umax[apv] : cal_->undershoot_[apv][strip];
    Uspread[apv] += cal_->undershoot_[apv][strip]*cal_->undershoot_[apv][strip]/nStrips;

    Bmean[apv] += cal_->baseline_[apv][strip]/nStrips;
    Bmin[apv] = Bmin[apv]<cal_->baseline_[apv][strip] ? Bmin[apv] : cal_->baseline_[apv][strip];
    Bmax[apv] = Bmax[apv]>cal_->baseline_[apv][strip] ? Bmax[apv] : cal_->baseline_[apv][strip];
    Bspread[apv] += cal_->baseline_[apv][strip]*cal_->baseline_[apv][strip]/nStrips;
  }
  
  // fill the mean, max, min, spread, ... histograms.
  for(int i=0;i<2;++i) {
    cal_->mean_amplitude_[i] = Amean[i];
    cal_->mean_tail_[i] = Tmean[i];
    cal_->mean_riseTime_[i] = Rmean[i];
    cal_->mean_timeConstant_[i] = Cmean[i];
    cal_->mean_turnOn_[i] = Omean[i];
    cal_->mean_maximum_[i] = Mmean[i];
    cal_->mean_undershoot_[i] = Umean[i];
    cal_->mean_baseline_[i] = Bmean[i];
    cal_->mean_smearing_[i] = Smean[i];
    cal_->mean_chi2_[i] = Kmean[i];
    cal_->min_amplitude_[i] = Amin[i];
    cal_->min_tail_[i] = Tmin[i];
    cal_->min_riseTime_[i] = Rmin[i];
    cal_->min_timeConstant_[i] = Cmin[i];
    cal_->min_turnOn_[i] = Omin[i];
    cal_->min_maximum_[i] = Mmin[i];
    cal_->min_undershoot_[i] = Umin[i];
    cal_->min_baseline_[i] = Bmin[i];
    cal_->min_smearing_[i] = Smin[i];
    cal_->min_chi2_[i] = Kmin[i];
    cal_->max_amplitude_[i] = Amax[i];
    cal_->max_tail_[i] = Tmax[i];
    cal_->max_riseTime_[i] = Rmax[i];
    cal_->max_timeConstant_[i] = Cmax[i];
    cal_->max_turnOn_[i] = Omax[i];
    cal_->max_maximum_[i] = Mmax[i];
    cal_->max_undershoot_[i] = Umax[i];
    cal_->max_baseline_[i] = Bmax[i];
    cal_->max_smearing_[i] = Smax[i];
    cal_->max_chi2_[i] = Kmax[i];
    cal_->spread_amplitude_[i] = sqrt(fabs(Aspread[i]-Amean[i]*Amean[i]));
    cal_->spread_tail_[i] = sqrt(fabs(Tspread[i]-Tmean[i]*Tmean[i]));
    cal_->spread_riseTime_[i] = sqrt(fabs(Rspread[i]-Rmean[i]*Rmean[i]));
    cal_->spread_timeConstant_[i] = sqrt(fabs(Cspread[i]-Cmean[i]*Cmean[i]));
    cal_->spread_turnOn_[i] = sqrt(fabs(Ospread[i]-Omean[i]*Omean[i]));
    cal_->spread_maximum_[i] = sqrt(fabs(Mspread[i]-Mmean[i]*Mmean[i]));
    cal_->spread_undershoot_[i] = sqrt(fabs(Uspread[i]-Umean[i]*Umean[i]));
    cal_->spread_baseline_[i] = sqrt(fabs(Bspread[i]-Bmean[i]*Bmean[i]));
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
  if ( cal_ ) { 
    if( cal_->isScan_ ){
      histo->Scale(1/16.);   
      edm::LogWarning(mlCommissioning_) 
	<< "CalibrationAlgorithm::correctDistribution: isScan_ == true!!! ";
    }
  }
}

// ----------------------------------------------------------------------------
//
TF1* CalibrationAlgorithm::fitPulse( TH1* histo, 
				     float rangeLow, 
				     float rangeHigh ) 
{
  if(!cal_) return nullptr;
  if(!histo) return nullptr;
  float noise = 4.;
  float N = round(histo->GetMaximum()/125.);
  float error = sqrt(2*N)*noise;

  for(int i=1;i<=histo->GetNbinsX();++i) {
    histo->SetBinError(i,error);
  }

  // let's help our fit with a bit of parameter guessing
  float maximum = histo->GetMaximum();
  float turn_on = histo->GetBinCenter((histo->FindFirstBinAbove(0.1 * maximum) - 2));

  if (cal_->deconv_) {

    deconv_fitter_->SetParameter(4, turn_on);

    if(rangeLow>rangeHigh)
      histo->Fit(deconv_fitter_,"QMR");
    else 
      histo->Fit(deconv_fitter_,"0QMR","",rangeLow,rangeHigh);

    return deconv_fitter_; 

  } 
  else {
    peak_fitter_->SetParameter(4, turn_on);
    if(rangeLow>rangeHigh)
      histo->Fit(peak_fitter_,"MRQ");    
    else 
      histo->Fit(peak_fitter_,"0QMR","",rangeLow,rangeHigh);
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
float CalibrationAlgorithm::turnOn(TF1* f){
   float max_amplitude = f->GetMaximum();
   float time = 5.;
   float base = f->GetMinimum(0,50);
   for( ; time < 50 && (f->Eval(time) - base) < 0.01 * (max_amplitude - base); time += 0.1) {}
   return time - 0.05;
}
