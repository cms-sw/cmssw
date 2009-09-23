#include "CalibTracker/SiStripLorentzAngle/interface/SymmetryFit.h"
#include <cmath>

TH1* SymmetryFit::symmetryChi2(const TH1* candidate, const std::pair<unsigned,unsigned> range) 
{
  std::cout << "symmetryChi2: "<< std::flush;
  std::pair<unsigned, unsigned> usable = range;
  while(usable.first  > 1                                  && candidate->GetBinError(usable.first-1)) --usable.first;
  while(usable.second < (unsigned)(candidate->GetNbinsX()) && candidate->GetBinError(usable.second+1)) ++usable.second;
  unsigned ndf = std::min(range.first-usable.first, usable.second-range.second);

  if( ndf < 10 || ndf < (range.second-range.first)/2 ) { std::cout << "bad range" << std::endl
								   << "usable: " << usable.first <<","<<usable.second << std::endl
								   << "range: " << range.first <<"," << range.second << std::endl; return 0;}

  SymmetryFit sf(candidate, range, ndf);
  int status = sf.fit();
  if(status) { delete sf.chi2_; sf.chi2_=0; }
  return sf.chi2_;
}

SymmetryFit::SymmetryFit(const TH1* h, const std::pair<unsigned,unsigned> r, const unsigned ndf)
  : symm_candidate_(h), 
    range_(r),
    ndf_(ndf),
    chi2_(0)
{
  makeChi2Histogram();
  fillchi2();
}

void SymmetryFit::makeChi2Histogram() 
{
  std::string XXname = name(symm_candidate_->GetName());
  unsigned Nbins = 2*( range_.second - range_.first ) + 3;
  double binwidth = symm_candidate_->GetBinWidth(1);
  double low = symm_candidate_->GetBinCenter(range_.first) - 3*binwidth/4;
  double up = symm_candidate_->GetBinCenter(range_.second) + 3*binwidth/4;
  chi2_ = new TH1F(XXname.c_str(),"", Nbins, low, up);
}

void SymmetryFit::fillchi2()
{
  for(int i=1;  i<=chi2_->GetNbinsX() ; ++i) {
    const unsigned L( range_.first-1+(i-1)/2 ), R( range_.first+i/2 );
    chi2_->SetBinContent( i, chi2( std::make_pair(L,R)) );
  }
}

float SymmetryFit::chi2(std::pair<unsigned,unsigned> point)
{
  float XX=0;
  unsigned i=ndf_;
  while(i-->0) {
    XX += chi2_element(point); 
    point.first--; 
    point.second++;
  }
  return XX;
}

float SymmetryFit::chi2_element(std::pair<unsigned,unsigned> range)
{
  float
    y0(symm_candidate_->GetBinContent(range.first)),
    y1(symm_candidate_->GetBinContent(range.second)),
    e0(symm_candidate_->GetBinError(range.first)),
    e1(symm_candidate_->GetBinError(range.second));
  
  return pow(y0-y1, 2)/(e0*e0+e1*e1);
}

int SymmetryFit::fit() {

  int status = chi2_->Fit("pol2","WQ");
  if(status) return status;
  double a = chi2_->GetFunction("pol2")->GetParameter(2);
  double b = chi2_->GetFunction("pol2")->GetParameter(1);
  double c = chi2_->GetFunction("pol2")->GetParameter(0);

  TF1* f = fitfunction();
  f->SetParameter(0, -0.5*b/a); f->SetParLimits(0, -0.5*b/a-0.01, -0.5*b/a+0.01);
  f->SetParameter(1, 1./sqrt(a)); f->SetParLimits(1, 0.9/sqrt(a), 1.1/sqrt(a));
  f->SetParameter(2, c-0.25*b*b/a); f->SetParLimits(2,-5, c);
  f->SetParameter(3, ndf_);  f->SetParLimits(3, ndf_,ndf_); //Fixed

  return chi2_->Fit(f,"W");
}

TF1* SymmetryFit::fitfunction() 
{
  TF1* f = new TF1("SymmetryFit","((x-[0])/[1])**2+[2]+0*[3]");
  f->SetParName(0,"middle");
  f->SetParName(1,"uncertainty");
  f->SetParName(2,"chi2");
  f->SetParName(3,"NDF");
  return f;
}
