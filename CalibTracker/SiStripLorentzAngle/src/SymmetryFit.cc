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
    std::cout << "|" << std::flush;
    SymmetryFit sf(candidate,range); 
    std::cout << ">" << std::flush;
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
    minDF_(2*(r.second-r.first)),
    range_(r),
    minmaxUsable_(findUsableMinMax()),
    ndf_( minmaxUsable_.first<minmaxUsable_.second ? minmaxUsable_.second-minmaxUsable_.first : 0),
    chi2_(0)
{
  std::cout << range_.first << "-" << range_.second << "," << minmaxUsable_.first<<"-"<<minmaxUsable_.second << std::flush;
  makeChi2Histogram();
  std::cout << ";" << std::flush;
  fillchi2();
  std::cout << ";" << std::flush;
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
    if( !test.first && err && i!=bins+1) test.first = i;
    else if( test.first && (!err||i==bins+1) ) {
      test.second = i-1;
      if( test.first  < range_.first-minDF_  && (test.second-test.first) > (bestL.second-bestL.first) ) bestL = test;
      if( test.second > range_.second+minDF_ && (test.second-test.first) > (bestR.second-bestR.first) ) bestR = test;
      test.first=test.second=0;
    }
  }
  if( bestL == notfound || bestR == notfound ) return std::make_pair(0,0);
  
  return std::make_pair( std::max( bestL.second > range_.second-1 ? 0 : range_.second-1-bestL.second,
				   bestR.first < range_.first+1 ? 0 : bestR.first-(range_.first+1)),
			 std::min( range_.first-1-bestL.first, bestR.second-range_.second) );
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

  std::vector<double> p = pol2_from_pol3(chi2_);
  if( !p.size() || 
      p[0] < chi2_->GetBinCenter(1) || 
      p[0] > chi2_->GetBinCenter(chi2_->GetNbinsX()))
    return 7;

  TF1* f = fitfunction();
  f->SetParameter(0, p[0]);  f->SetParLimits(0, p[0], p[0]);
  f->SetParameter(1, p[1]);  f->SetParLimits(1, p[1], p[1]);
  f->SetParameter(2, p[2]);  f->SetParLimits(2, p[2], p[2]);
  f->SetParameter(3, ndf_);  f->SetParLimits(3, ndf_,ndf_); //Fixed
  chi2_->Fit(f,"WQ");
  return 0;
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


std::vector<double> SymmetryFit::pol2_from_pol2(TH1* hist) {
  std::vector<double> v;

  int status = hist->Fit("pol2","WQ");
  if(!status) {
    std::vector<double> p;
    p.push_back(hist->GetFunction("pol2")->GetParameter(0));
    p.push_back(hist->GetFunction("pol2")->GetParameter(1));
    p.push_back(hist->GetFunction("pol2")->GetParameter(2));
    if(p[2]>0) {
      v.push_back( -0.5*p[1]/p[2] );
      v.push_back( 1./sqrt(p[2]) );
      v.push_back( p[0]-0.25*p[1]*p[1]/p[2] );
    }
  }
  return v;
}

std::vector<double> SymmetryFit::pol2_from_pol3(TH1* hist) {
  std::vector<double> v;

  int status = hist->Fit("pol3","WQ");
  if(!status) {
    std::vector<double> p;
    p.push_back(hist->GetFunction("pol3")->GetParameter(0));
    p.push_back(hist->GetFunction("pol3")->GetParameter(1));
    p.push_back(hist->GetFunction("pol3")->GetParameter(2));
    p.push_back(hist->GetFunction("pol3")->GetParameter(3));
    double radical = p[2]*p[2] - 3*p[1]*p[3] ;
    if(radical>0) {
      double x0 = ( -p[2] + sqrt(radical) ) / ( 3*p[3] ) ;
      v.push_back( x0 );
      v.push_back( pow( radical, -0.25) );
      v.push_back( p[0] + p[1]*x0 + p[2]*x0*x0 + p[3]*x0*x0*x0 );
    }
  }
  return v;
}
