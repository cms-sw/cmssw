#include "CalibTracker/SiStripLorentzAngle/interface/SymmetryFit.h"
#include <cmath>
#include "boost/foreach.hpp"

TH1* SymmetryFit::symmetryChi2(std::string basename, const std::vector<TH1*>& candidates, const std::pair<unsigned,unsigned> range)
{
  TH1* fake = (TH1*)(candidates[0]->Clone(basename.c_str())); 
  fake->Reset();
  SymmetryFit combined(fake,range);
  delete fake;

  BOOST_FOREACH(const TH1* candidate, candidates) {
    SymmetryFit sf(candidate,range); 
    combined+=sf; 
    delete sf.chi2_; 
  }

  int status = combined.fit();
  if(status) { delete combined.chi2_; combined.chi2_=0;}
  return combined.chi2_;
}

TH1* SymmetryFit::symmetryChi2(const TH1* candidate, const std::pair<unsigned,unsigned> range) 
{
  SymmetryFit sf(candidate, range);
  int status = sf.fit();
  if(status) { delete sf.chi2_; sf.chi2_=0; }
  return sf.chi2_;
}

SymmetryFit::SymmetryFit(const TH1* h, const std::pair<unsigned,unsigned> r)
  : symm_candidate_(h), 
    minDF_(10),
    range_(r),
    minmaxUsable_(findUsableMinMax()),
    ndf_(minmaxUsable_.second-minmaxUsable_.first+1),
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

std::pair<unsigned,unsigned> SymmetryFit::findUsableMinMax()
{
  std::pair<unsigned, unsigned> bestL(0,0), bestR(0,0),test(0,0), notfound(0,0);
  unsigned bins = symm_candidate_->GetNbinsX();
  for(unsigned i = 1; i <= bins+1; i++) {
    float err = symm_candidate_->GetBinError(i);
    if( !test.first && err ) test.first = i;
    if( test.first && !test.second && (!err||i==bins+1) ) {
      test.second = i-1;
      if( test.first  < range_.first-minDF_  && (test.second-test.first) > (bestL.second-bestL.first) ) bestL = test;
      if( test.second > range_.second+minDF_ && (test.second-test.first) > (bestR.second-bestR.first) ) bestR = test;
      test.first=test.second=0;
    }
  }
  if( bestL == notfound || bestR == notfound ) return std::make_pair(0,0);
  
  return std::make_pair( std::max( bestL.second > range_.second-1 ? 0 : range_.second-1-bestL.second,
				   bestR.first < range_.first+1 ? 0 : bestR.first-(range_.first+1)),
			 std::min( range_.first-1-bestL.first, bestR.second+1-range_.second) );
}

void SymmetryFit::fillchi2()
{
  if(ndf_<minDF_) return;

  for(int i=1;  i<=chi2_->GetNbinsX() ; ++i) {
    const unsigned L( range_.first-1+(i-1)/2 ), R( range_.first+i/2 );
    chi2_->SetBinContent( i, chi2( std::make_pair(L,R)) );
  }
}

float SymmetryFit::chi2(std::pair<unsigned,unsigned> point)
{
  point.first -= minmaxUsable_.first;
  point.second += minmaxUsable_.first;
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
  if( a<0  || 
      -0.5*b/a < chi2_->GetBinCenter(1) || 
      -0.5*b/a > chi2_->GetBinCenter(chi2_->GetNbinsX()))
    return 7;

  TF1* f = fitfunction();
  f->SetParameter(0, -0.5*b/a);
  f->SetParameter(1, 1./sqrt(a));
  f->SetParameter(2, c-0.25*b*b/a);
  f->SetParameter(3, ndf_);  f->SetParLimits(3, ndf_,ndf_); //Fixed

  return chi2_->Fit(f,"WQ");
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
