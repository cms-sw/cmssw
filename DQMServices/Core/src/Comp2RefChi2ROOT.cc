#include "DQMServices/Core/interface/Comp2RefChi2ROOT.h"

#include <iostream>

#include <TMath.h>
#include <TH1F.h>

using std::cout; using std::endl; using std::cerr;

// true if test cannot run
bool Comp2RefChi2ROOT::isInvalid(const TH1F * const h)
{
  if(hasNullReference())return true;
  if(!h) return true;

  //check dimensions
  if (h->GetDimension()!=1 || ref_->GetDimension()!=1){
    cerr << " Comp2RefChi2ROOT error: Histograms must be 1-D\n " << endl;;
    return true;
  }

  TAxis *axis1 = h->GetXaxis();
  TAxis *axis2 = ref_->GetXaxis();
  nbins1 = axis1->GetNbins();
  nbins2 = axis2->GetNbins();
  
  //check number of channels
  if (nbins1 != nbins2){
    cerr << " Comp2RefChi2ROOT error: different number of channels! (" << nbins1 
	 << ", " << nbins2 << ") " << endl;
    return true;
  }

  // if here, everything is good
  return false;
}

float Comp2RefChi2ROOT::runTest(const TH1F * const h)
{
  resetResults();

  if(isInvalid(h))return -1;

  // copy & past from TH1::Chi2TestX, till we have public method
  // for all of chi2, NDF and probability variables...
  
  Int_t i, i_start, i_end;
  float chi2 = 0;
  int ndof = 0;
  int constraint = 0;

  TAxis *axis1 = h->GetXaxis();
  //see options  
  i_start = 1;
  i_end = nbins1;
  //  if (fXaxis.TestBit(TAxis::kAxisRange)) {
  i_start = axis1->GetFirst();
  i_end   = axis1->GetLast();
  //  }
  ndof = i_end-i_start+1-constraint;

  //Compute the normalisation factor
  Double_t sum1=0, sum2=0;
  for (i=i_start; i<=i_end; i++){
    sum1 += h->GetBinContent(i);
    sum2 += ref_->GetBinContent(i);
  }

  //check that the histograms are not empty
  if (sum1 == 0 || sum2 == 0){
    cerr << " Comp2RefChi2ROOT error: one of the histograms is empty" << endl;;
    return -1;
  }
  
  Double_t bin1, bin2, err1, err2, temp;
  for (i=i_start; i<=i_end; i++){
    bin1 = h->GetBinContent(i)/sum1;
    bin2 = ref_->GetBinContent(i)/sum2;
    if (bin1 ==0 && bin2==0){
      --ndof; //no data means one less degree of freedom
    } else {
      
      temp  = bin1-bin2;        
      err1=h->GetBinError(i);
      err2=ref_->GetBinError(i);
      if (err1 == 0 && err2 == 0){
	cerr << " Comp2RefChi2ROOT error: bins with non-zero content and zero error"
	     << endl;
	return -1;
      }
      err1*=err1;
      err2*=err2;
      err1/=sum1*sum1;
      err2/=sum2*sum2;
      chi2 +=temp*temp/(err1+err2);
    }
  }

  chi2_ = chi2;
  Ndof_ = ndof;
  return TMath::Prob(0.5*chi2, Int_t(0.5*ndof));
}

void Comp2RefChi2ROOT::resetResults(void)
{
  Ndof_ = 0; chi2_ = -1; nbins1 = nbins2 = -1;
}
