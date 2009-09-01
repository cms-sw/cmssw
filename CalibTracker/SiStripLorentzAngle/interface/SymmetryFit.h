#ifndef CalibTracker_SiStripLorentzAngle_SymmetryFit
#define CalibTracker_SiStripLorentzAngle_SymmetryFit

#include <vector>
#include <cmath>
#include <TGraph.h>
#include <TF1.h>
#include <TProfile.h>

namespace sistrip {

class SymmetryFit {

 public:
  static SymmetryFit fromTProfile(TProfile* p) {
    std::vector<float> x,y,e;
    for(int i=1; i <= p->GetNbinsX() ; ++i) {
      if(p->GetBinEntries(i) < 3) {
	x.clear(); y.clear(); e.clear(); i=1;
	if(p->GetNbinsX() %2) return SymmetryFit(x,y,e);
	p->Rebin();
      }
      x.push_back(p->GetBinCenter(i));
      y.push_back(p->GetBinContent(i));
      e.push_back(p->GetBinError(i));
    }
    return SymmetryFit(x,y,e);
  }

  SymmetryFit(std::vector<float>& x, std::vector<float>& y, std::vector<float>& e) 
    : x(x), y(y), e(e) {
    if(x.size()!=y.size() ||
       x.size()!=e.size() ||
       x.size()<10) {
      status_=4;
    } else fit();
  }
    
  std::pair<float,float> result() const { return std::make_pair(-0.5*p1_/p2_, 1/sqrt(p2_));}
  float minChi2() const {float x = result().first; return p0_ + p1_*x + p2_*x*x ;}
  unsigned NDF() const {return ndf_;}
  int fitStatus() const {return status_;}
    
 private:

  const std::vector<float>& x,y,e;
  std::vector<float> xX2, X2;
  unsigned ndf_;
  int status_;
  float p0_,p1_,p2_;

  void fit() {
    fill_chi2_array(std::make_pair(x.size()/5,(4*x.size())/5));
    unsigned imin = x.size()/5+(std::min_element(X2.begin(),X2.end())-X2.begin())/2;
    if( std::min(imin-3, x.size()-imin-3) > 3)
      ndf_ = fill_chi2_array(std::make_pair(imin-3,imin+3));
    else if (std::min(imin-2, x.size()-imin-2) > 2)
      ndf_ = fill_chi2_array(std::make_pair(imin-2,imin+2));
    else {status_=4; return;}
    
    TGraph graph(X2.size(), &(xX2[0]), &(X2[0]));
    status_ = graph.Fit("pol2","Q");
    p0_ = graph.GetFunction("pol2")->GetParameter(0);
    p1_ = graph.GetFunction("pol2")->GetParameter(1);
    p2_ = graph.GetFunction("pol2")->GetParameter(2);
  }

  unsigned fill_chi2_array(std::pair<unsigned,unsigned> window) {
    xX2.clear();  X2.clear();
    const unsigned 
      ndf (std::min( window.first, x.size()-window.second-1) ),
      imin ( 2*window.first-1 ),
      imax ( 2*window.second +2 );

    for(unsigned i=imin;  i<imax ; ++i) {
      const unsigned L( (i-1)/2 ), R( (i+2)/2 );
      xX2.push_back( (x[L]+x[R]) / 2   );
      X2.push_back( chi2( ndf, std::make_pair(L,R)) );
    }
    return ndf;
  }

  float chi2(unsigned ndf, std::pair<unsigned, unsigned> ij) const {
    float X2 = 0;
    while( 0 < ndf-- )  X2 +=  chi2_ij( ij.first++, ij.second--);
    return X2;
  }

  float chi2_ij(unsigned i, unsigned j) const 
  { return pow( y.at(i) - y.at(j), 2)  /  ( pow( e.at(i),2) + pow( e.at(j),2) );}

  void print_chi2_array() {
    for(unsigned i=0; i< X2.size(); ++i) {
      std::cout << xX2[i] << ",\t" << X2[i] << std::endl;
    }
  }

};

}
#endif
