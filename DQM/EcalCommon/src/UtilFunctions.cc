/*!
  \file UtilFunctions.cc
  \brief Ecal Monitor Utility functions
  \author Dongwook Jang
  \version $Revision: 1.7 $
  \date $Date: 2012/04/27 13:46:04 $
*/

#include "DQM/EcalCommon/interface/UtilFunctions.h"
#include <cmath>
#include <iostream>

#include "TH1F.h"
#include "TProfile.h"
#include "TH2.h"
#include "TClass.h"
#include "TAxis.h"
#include "DQMServices/Core/interface/MonitorElement.h"


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

  void shift(TH1 *h, Directions d, int bins)
  {
    if(!bins || !h) return;
    if(h->GetXaxis()->IsVariableBinSize()) return;

    if(bins < 0){
      bins = -bins;
      d = d==kRight ? kLeft : kRight;
    }

    if(!h->GetSumw2()) h->Sumw2();
    int nBins = h->GetXaxis()->GetNbins();
    if(bins >= nBins){
      h->Reset();
      return;
    }

    // the first bin goes to underflow
    // each bin moves to the right

    int firstBin, lastBin, bound, increment;
    switch(d){
    case kRight:
      firstBin = nBins + 1;
      lastBin = 0;
      bound = bins;
      increment = -1;
      break;
    case kLeft:
      firstBin = 0;
      lastBin = nBins + 1;
      bound = nBins - bins + 1;
      increment = 1;
      break;
    default:
      return;
    }

    int shift = increment * bins;

    if( h->IsA() == TClass::GetClass("TProfile") ){

      TProfile *p = static_cast<TProfile *>(h);

      // by shifting n bin to the left, the number of entries are
      // reduced by the number in n bins including the underflow bin.
      double nentries = p->GetEntries();
      for(int i = firstBin; i != firstBin + shift; i += increment) nentries -= p->GetBinEntries(i);
      p->SetEntries(nentries);

      TArrayD* sumw2 = p->GetSumw2();

      for(int i = firstBin; i != bound; i += increment){
	// GetBinContent returns binContent/binEntries
	p->SetBinContent( i, p->GetBinContent( i+shift ) * p->GetBinEntries( i+shift ) );
	p->SetBinEntries( i, p->GetBinEntries( i+shift ) );
	sumw2->SetAt( sumw2->GetAt( i+shift ), i );
      }

      for(int i = bound; i != lastBin + increment; i += increment){
	p->SetBinContent( i, 0 );
	p->SetBinEntries( i, 0 );
	sumw2->SetAt( 0., i );
      }

    }else if( h->InheritsFrom("TH2") ){

      TH2 *h2 = static_cast<TH2 *>(h);
      int nBinsY = h2->GetYaxis()->GetNbins();

      // assumes sum(binContent) == entries
      double nentries = h2->GetEntries();
      for(int i = firstBin; i != firstBin + shift; i += increment)
	for(int j=0 ; j<=nBinsY+1 ; j++) nentries -= h2->GetBinContent(i,j);

      h2->SetEntries(nentries);

      for(int i = firstBin; i != bound; i += increment)
	for(int j = 0; j <= nBinsY + 1; j++)
	  h2->SetBinContent( i, j, h2->GetBinContent(i+shift, j) );

      for(int i = bound; i != lastBin + increment; i += increment)
	for(int j = 0; j <= nBinsY + 1; j++)
	  h2->SetBinContent( i, j, 0 );
	  
    }else if( h->InheritsFrom("TH1") ){ // any other histogram class

      // assumes sum(binContent) == entries
      double nentries = h->GetEntries();
      for(int i = firstBin; i != firstBin + shift; i += increment) nentries -= h->GetBinContent(i);

      h->SetEntries(nentries);

      for(int i = firstBin; i != bound; i += increment)
	h->SetBinContent( i, h->GetBinContent(i+shift) );

      for(int i = bound; i != lastBin + increment; i += increment)
	h->SetBinContent( i, 0 );
    }


  }

  void shift2Right(TH1* h, int bins)
  {
    shift(h, kRight, bins);
  }

  void shift2Left(TH1* h, int bins)
  {
    shift(h, kLeft, bins);
  }

  // shift axis of histograms keeping the bin contents

  void shiftAxis(TH1 *h, Directions d, double shift)
  {
    if( !h ) return;
    TAxis *xax = h->GetXaxis();
    if( h->GetXaxis()->IsVariableBinSize() ) return;

    double xmax = xax->GetXmax();
    double xmin = xax->GetXmin();
    int nbins = xax->GetNbins();

    if(d == kRight)
      xax->Set(nbins, xmin - shift, xmax - shift);
    else if(d == kLeft)
      xax->Set(nbins, xmin + shift, xmax + shift);
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
	if(mean < 0) return;
	rms = std::sqrt(mean);
      }
      else {
	mean = ((TH2F*)cur)->GetEntries();
	if(mean < 0) return;
	rms = std::sqrt(mean);
      }
      float nxybins = ((TH2F*)cur)->GetNbinsX()*((TH2F*)cur)->GetNbinsY();
      if(nxybins < 1.) nxybins = 1.;
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

