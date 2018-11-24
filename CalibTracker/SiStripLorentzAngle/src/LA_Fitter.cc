#include "CalibTracker/SiStripLorentzAngle/interface/LA_Filler_Fitter.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cmath>
#include <boost/algorithm/string/erase.hpp>
#include <TF1.h>

void LA_Filler_Fitter::
fit_width_profile(Book& book) {
  for(Book::iterator it = book.begin(".*"+method(WIDTH)); it!=book.end(); ++it) {
    it->second->SetTitle("Mean Cluster Width;tan#theta_{t}");
    TH1* const p = it->second;
    if(p->GetEntries() < 400) { delete p; book[it->first]=nullptr; continue;}
    p->SetTitle(";tan#theta_{t};");
    const float min = p->GetMinimum();
    const float max = p->GetMaximum();
    float xofmin = p->GetBinCenter(p->GetMinimumBin()); if( xofmin>0.0 || xofmin<-0.15) xofmin = -0.05;
    const float xofmax = p->GetBinCenter(p->GetMaximumBin());

    TF1* const fit = new TF1("LA_profile_fit","[2]*(TMath::Abs(x-[0]))+[1]",-1,1);
    fit->SetParLimits(0,-0.15,0.01);
    fit->SetParLimits(1, 0.6*min, 1.25*min );
    fit->SetParLimits(2,0.1,10);
    fit->SetParameters( xofmin, min, (max-min) / fabs( xofmax - xofmin ) );

    int badfit = p->Fit(fit,"IEQ","",-.5,.3);
    if( badfit ) badfit = p->Fit(fit,"IEQ","", -.46,.26);
    if( badfit ) { book.erase(it); }
  }
}

void LA_Filler_Fitter::
make_and_fit_symmchi2(Book& book) {
  for(Book::iterator it = book.begin(".*_all"); it!=book.end(); ++it) {
    const std::string base = boost::erase_all_copy(it->first,"_all");

    std::vector<Book::iterator> rebin_hists;              
    const Book::iterator&    all = it;	                             rebin_hists.push_back(all);   
    Book::iterator     w1 = book.find(base+"_w1");           rebin_hists.push_back(w1);    
    Book::iterator var_w2 = book.find(base+method(AVGV2,false)); rebin_hists.push_back(var_w2);
    Book::iterator var_w3 = book.find(base+method(AVGV3,false)); rebin_hists.push_back(var_w3);

    const unsigned rebin = std::max( var_w2==book.end() ? 0 : find_rebin(var_w2->second), 
				     var_w3==book.end() ? 0 : find_rebin(var_w3->second) );
    for(const auto& it : rebin_hists) if(it!=book.end()) it->second->Rebin( rebin>1 ? rebin<7 ? rebin : 6 : 1);

    TH1* const prob_w1 = w1==book.end()     ? nullptr : subset_probability( base+method(PROB1,false) ,w1->second,all->second);
    TH1* const rmsv_w2 = var_w2==book.end() ? nullptr :        rms_profile( base+method(RMSV2,false), (TProfile*const)var_w2->second);
    TH1* const rmsv_w3 = var_w3==book.end() ? nullptr :        rms_profile( base+method(RMSV3,false), (TProfile*const)var_w3->second);
    
    std::vector<TH1*> fit_hists;
    if(prob_w1) {
      book.book(base+method(PROB1,false),prob_w1);
      fit_hists.push_back(prob_w1);  prob_w1->SetTitle("Width==1 Probability;tan#theta_{t}-(dx/dz)_{reco}");
    }
    if(var_w2!=book.end())  {
      book.book(base+method(RMSV2,false),rmsv_w2);
      fit_hists.push_back(var_w2->second);   var_w2->second->SetTitle("Width==2 Mean Variance;tan#theta_{t}-(dx/dz)_{reco}");
      fit_hists.push_back(rmsv_w2);                 rmsv_w2->SetTitle("Width==2 RMS Variance;tan#theta_{t}-(dx/dz)_{reco}");
    }
    if(var_w3!=book.end())  {
      book.book(base+method(RMSV3,false),rmsv_w3);
      fit_hists.push_back(var_w3->second);   var_w3->second->SetTitle("Width==3 Mean Variance;tan#theta_{t}-(dx/dz)_{reco}");
      fit_hists.push_back(rmsv_w3);                 rmsv_w3->SetTitle("Width==3 RMS Variance;tan#theta_{t}-(dx/dz)_{reco}");
    }

    if(fit_hists.empty()) continue;
    const unsigned bins = fit_hists[0]->GetNbinsX();
    const unsigned guess = fit_hists[0]->FindBin(0);
    const std::pair<unsigned,unsigned> range(guess-bins/30,guess+bins/30);

    for(auto const& hist : fit_hists) {
      TH1*const chi2 = SymmetryFit::symmetryChi2(hist,range);
      if(chi2) {book.book(chi2->GetName(),chi2); chi2->SetTitle("Symmetry #chi^{2};tan#theta_{t}-(dx/dz)_{reco}");}
    }
  }
}

unsigned LA_Filler_Fitter::
find_rebin(const TH1* const hist) {
  const double mean = hist->GetMean();
  const double rms = hist->GetRMS();
  const int begin = std::min(                1, hist->GetXaxis()->FindFixBin(mean-rms));
  const int end   = std::max(hist->GetNbinsX(), hist->GetXaxis()->FindFixBin(mean+rms)) + 1;
  unsigned current_hole(0), max_hole(0);
  for(int i=begin; i<end; i++) {
    if(!hist->GetBinError(i)) current_hole++;
    else if(current_hole) {max_hole = std::max(current_hole,max_hole); current_hole=0;}
  }
  return max_hole+1;
}

TH1* LA_Filler_Fitter::
rms_profile(const std::string name, const TProfile* const prof) {
  const int bins = prof->GetNbinsX();
  TH1* const rms = new TH1F(name.c_str(),"",bins, prof->GetBinLowEdge(1),  prof->GetBinLowEdge(bins) + prof->GetBinWidth(bins) );
  for(int i = 1; i<=bins; i++) {
    const double Me = prof->GetBinError(i);
    const double neff = prof->GetBinEntries(i); //Should be prof->GetBinEffectiveEntries(i);, not availible this version ROOT.  This is only ok for unweighted fills
    rms->SetBinContent(i, Me*sqrt(neff) );
    rms->SetBinError(i, Me/sqrt(2) );
  }
  return rms;
}

TH1* LA_Filler_Fitter::
subset_probability(const std::string name, const TH1* const subset, const TH1* const total) {
  const int bins = subset->GetNbinsX();
  TH1* const prob = new TH1F(name.c_str(),"",bins, subset->GetBinLowEdge(1),  subset->GetBinLowEdge(bins) + subset->GetBinWidth(bins) );
  for(int i = 1; i<=bins; i++) {
    const double s = subset->GetBinContent(i);
    const double T = total->GetBinContent(i);
    const double B = T-s;

    const double p = T? s/T : 0;
    const double perr = T? ( (s&&B)? sqrt(s*s*B+B*B*s)/(T*T) : 1/T ) : 0;

    prob->SetBinContent(i,p);
    prob->SetBinError(i,perr);
  }  
  return prob;
}
