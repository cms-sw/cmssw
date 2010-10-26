#ifndef UtilFunctions_H
#define UtilFunctions_H

/*!
  \file UtilFunctions.h
  \brief Ecal Monitor Utility functions
  \author Dongwook Jang
  \version $Revision: 1.1 $
  \date $Date: 2010/02/08 21:33:31 $
*/

#include <cmath>
#include <iostream>

#include "TH1F.h"
#include "TProfile.h"
#include "TClass.h"


namespace ecaldqm {

  // functions implemented here are not universal in the sense that
  // the input variables are changed due to efficiency of memory usage.


  // calculated time intervals and bin locations for time varing profiles

  void calcBins(int binWidth, int divisor, long int start_time, long int last_time, long int current_time,
		long int & binDiff, long int & diff) {

    // changing arguments : binDiff, diff

    // binWidth : time interval
    // divisor : time unit - for minute case divisor is 60 and for hour case 3600
    // start_time : initial time when the job started
    // last_time : the last updated time before calling the current "analyze" function
    // current_time : the current time inside "analyze" fucntion
    // binDiff : the bin difference for the current time compared to the bin location of the last time
    // diff : time difference between the current time and the last time

    long int diff_current_start = current_time - start_time;
    long int diff_last_start = last_time - start_time;

    // --------------------------------------------------
    // Calculate time interval and bin width
    // --------------------------------------------------

    binDiff = diff_current_start/divisor/binWidth - diff_last_start/divisor/binWidth;
    diff = (current_time - last_time)/divisor;

    if(diff >= binWidth) {
      while(diff >= binWidth) diff -= binWidth;
    }

  } // calcBins



  // shift bins in TProfile to the right

  void shift2Right(TProfile* p, int bins){

    // bins : how many bins need to be shifted

    if(bins <= 0) return;

    if(!p->GetSumw2()) p->Sumw2();
    int nBins = p->GetXaxis()->GetNbins();

    // by shifting n bin to the right, the number of entries are
    // reduced by the number in n bins including the overflow bin.
    double nentries = p->GetEntries();
    for(int i=0; i<bins; i++) nentries -= p->GetBinEntries(i);
    p->SetEntries(nentries);
  
    // the last bin goes to overflow
    // each bin moves to the right

    TArrayD* sumw2 = p->GetSumw2();

    for(int i=nBins+1; i>bins; i--) {
      // GetBinContent return binContent/binEntries
      p->SetBinContent(i, p->GetBinContent(i-bins)*p->GetBinEntries(i-bins));
      p->SetBinEntries(i,p->GetBinEntries(i-bins));
      sumw2->SetAt(sumw2->GetAt(i-bins),i);
    }
    
  }


  // shift bins in TProfile to the left

  void shift2Left(TProfile* p, int bins){

    if(bins <= 0) return;

    if(!p->GetSumw2()) p->Sumw2();
    int nBins = p->GetXaxis()->GetNbins();

    // by shifting n bin to the left, the number of entries are
    // reduced by the number in n bins including the underflow bin.
    double nentries = p->GetEntries();
    for(int i=0; i<bins; i++) nentries -= p->GetBinEntries(i);
    p->SetEntries(nentries);
  
    // the first bin goes to underflow
    // each bin moves to the right

    TArrayD* sumw2 = p->GetSumw2();

    for(int i=0; i<=nBins+1-bins; i++) {
      // GetBinContent return binContent/binEntries
      p->SetBinContent(i, p->GetBinContent(i+bins)*p->GetBinEntries(i+bins));
      p->SetBinEntries(i,p->GetBinEntries(i+bins));
      sumw2->SetAt(sumw2->GetAt(i+bins),i);
    }

  }


  // get mean and rms of Y values from TProfile

  void getAverageFromTProfile(TProfile* p, double& mean, double& rms) {

    // changing arguments : mean, rms
    mean = rms = 0.0;

    if(!p) return;

    int nbins = p->GetXaxis()->GetNbins();
    double y = 0.0;
    double y2 = 0.0;
    for(int i=1; i<=nbins; i++){
      y += p->GetBinContent(i);
      y2 += y*y;
    }
    mean = y/nbins;
    rms = std::sqrt(y2/nbins - mean*mean);

  } // getAverageFromTProfile


  // get mean and rms based on two histograms' difference

  void getMeanRms(TObject* pre, TObject* cur, double& mean, double& rms) {

    // changing arguments : mean, rms

    mean = rms = 0.0;

    if(!cur) return;

    TString name(cur->IsA()->GetName());

    if(name.Contains("TProfile")) {
      getAverageFromTProfile((TProfile*)cur,mean,rms);
    }
    else if(name.Contains("TH2")) {
      if(pre) {
	mean = ((TH2F*)cur)->GetEntries() - ((TH2F*)pre)->GetEntries();
	rms = std::sqrt(mean);
      }
      else {
	mean = ((TH2F*)cur)->GetEntries();
	rms = std::sqrt(mean);
      }
      float nxybins = ((TH2F*)cur)->GetNbinsX()*((TH2F*)cur)->GetNbinsY();
      mean /= nxybins;
      rms /= nxybins;
    }
    else if(name.Contains("TH1")) {
      if(pre) {
	((TH1F*)pre)->Sumw2();
	((TH1F*)pre)->Add((TH1F*)pre,(TH1F*)cur,-1,1);
	mean = ((TH1F*)pre)->GetMean();
	rms = ((TH1F*)pre)->GetRMS();
      }
      else {
	mean = ((TH1F*)cur)->GetMean();
	rms = ((TH1F*)cur)->GetRMS();
      }
    }

  } // getMeanRms

  TObject* cloneIt(MonitorElement* me, std::string histo) {

    // The cloned object, ret should be deleted after using it.

    TObject* ret = 0;

    if(!me) return ret;

    std::string title = histo + " Clone";
    ret = (TObject*) (me->getRootObject()->Clone(title.c_str()));

    return ret;
  }


}

#endif
