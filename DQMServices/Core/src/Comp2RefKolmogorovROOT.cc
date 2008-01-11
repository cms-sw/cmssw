#include "DQMServices/Core/interface/Comp2RefKolmogorovROOT.h"

#include <iostream>
#include "TMath.h"

using std::cout; using std::endl; using std::cerr;

// true if test cannot run
bool Comp2RefKolmogorovROOT::isInvalid(const TH1F * const h)
{
  if(hasNullReference())return true;
  if(!h) return true;
  // Check consistency of dimensions
  if (h->GetDimension() != 1 || ref_->GetDimension() != 1) {
    cerr << " KolmogorovTest error: Histograms must be 1-D\n";
    return true;
  }

  TAxis *axis1 = h->GetXaxis();
  TAxis *axis2 = ref_->GetXaxis();
  ncx1   = axis1->GetNbins();
  ncx2   = axis2->GetNbins();
  
  // Check consistency in number of channels
  if (ncx1 != ncx2) {
    cerr << " KolmogorovTest error: different number of channels! (" << ncx1 
	 << ", " << ncx2 << ") " << endl;
    return true;
  }

  // Check consistency in channel edges
  Double_t diff1 = TMath::Abs(axis1->GetXmin() - axis2->GetXmin());
  Double_t diff2 = TMath::Abs(axis1->GetXmax() - axis2->GetXmax());
  if (diff1 > difprec || diff2 > difprec) {
    cerr << " KolmogorovTest error: histograms with different binning";
    return true;
  }

  // if here, everything is good
  return false;

}

const Double_t Comp2RefKolmogorovROOT::difprec = 1e-5;

float Comp2RefKolmogorovROOT::runTest(const TH1F * const h)
{
  if(isInvalid(h))return -1;

  // copy & past from TH1::KolmogorovTest, till we have public method
  // for all of chi2, NDF and probability variables...
  Bool_t afunc1 = kFALSE;
  Bool_t afunc2 = kFALSE;
  Double_t sum1 = 0, sum2 = 0;
  Double_t ew1, ew2, w1 = 0, w2 = 0;
  Int_t bin;
  for (bin=1;bin<=ncx1;bin++) {
    sum1 += h->GetBinContent(bin);
    sum2 += ref_->GetBinContent(bin);
    ew1   = h->GetBinError(bin);
    ew2   = ref_->GetBinError(bin);
    w1   += ew1*ew1;
    w2   += ew2*ew2;
  }
   if (sum1 == 0) {
     cerr << "KolmogorovTest error: Histogram " << h->GetName() 
	  << " integral is zero\n";
      return -1;
   }
   if (sum2 == 0) {
     cerr << "KolmogorovTest error: Histogram " << ref_->GetName() 
	  << " integral is zero\n";
     return -1;
   }

   Double_t tsum1 = sum1;
   Double_t tsum2 = sum2;
   tsum1 += h->GetBinContent(0);
   tsum2 += ref_->GetBinContent(0);
   tsum1 += h->GetBinContent(ncx1+1);
   tsum2 += ref_->GetBinContent(ncx1+1);
   
   // Check if histograms are weighted.
   // If number of entries = number of channels, probably histograms were
   // not filled via Fill(), but via SetBinContent()
   Double_t ne1 = h->GetEntries();
   Double_t ne2 = ref_->GetEntries();
   // look at first histogram
   Double_t difsum1 = (ne1-tsum1)/tsum1;
   Double_t esum1 = sum1;
   if (difsum1 > difprec && Int_t(ne1) != ncx1) {
     if (h->GetSumw2N() == 0) {
       cout << " KolmogorovTest warning: Weighted events and no Sumw2 for "
	    << h->GetName() << endl;
     } else {
       esum1 = sum1*sum1/w1;  //number of equivalent entries
     }
   }
   // look at second histogram
   Double_t difsum2 = (ne2-tsum2)/tsum2;
   Double_t esum2   = sum2;
   if (difsum2 > difprec && Int_t(ne2) != ncx1) {
     if (ref_->GetSumw2N() == 0) {
       cout << " KolmogorovTest warning: Weighted events and no Sumw2 for "
	    << ref_->GetName() << endl;
     } else {
       esum2 = sum2*sum2/w2;  //number of equivalent entries
     }
   }

   Double_t s1 = 1/tsum1;
   Double_t s2 = 1/tsum2;
   
   // Find largest difference for Kolmogorov Test
   Double_t dfmax =0, rsum1 = 0, rsum2 = 0;

   // use underflow bin
   Int_t first = 0; // 1 
   // use overflow bin
   Int_t last  = ncx1+1; // ncx1
   for (bin=first;bin<=last;bin++) {
     rsum1 += s1*h->GetBinContent(bin);
     rsum2 += s2*ref_->GetBinContent(bin);
     dfmax = TMath::Max(dfmax,TMath::Abs(rsum1-rsum2));
   }

   // Get Kolmogorov probability
   Double_t z = 0;
   if (afunc1)      z = dfmax*TMath::Sqrt(esum2);
   else if (afunc2) z = dfmax*TMath::Sqrt(esum1);
   else             z = dfmax*TMath::Sqrt(esum1*esum2/(esum1+esum2));
   
   // This numerical error condition should never occur:
   if (TMath::Abs(rsum1-1) > 0.002)
     cout << " KolmogorovTest warning: Numerical problems with histogram "
	  << h->GetName() << endl;
   if (TMath::Abs(rsum2-1) > 0.002) 
     cout << " KolmogorovTest warning: Numerical problems with histogram "
	  << ref_->GetName() << endl;

   return TMath::KolmogorovProb(z);
     
}
