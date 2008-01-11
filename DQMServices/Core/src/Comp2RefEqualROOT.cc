#include "DQMServices/Core/interface/Comp2RefEqualROOT.h"

#include <iostream>

using std::string; using std::cerr; using std::endl;

// Note: runTest was supposed to be a method of the (template) class Comp2RefEqual;
// This caused problems with the ROOT histograms, because operator== is not defined
// for TH1F, TH2F, TH3F, etc; So, implementing everything here for now
// christos, 01-FEB-06

// run the test (result: [0, 1] or <0 for failure)
float Comp2RefEqualStringROOT::runTest(const string * const t)
{
  if(Comp2RefEqual<string>::isInvalid(t))return -1;

  if(*t == *ref_)
    return 1;
  else
    return 0;
}

// run the test (result: [0, 1] or <0 for failure)
float Comp2RefEqualIntROOT::runTest(const int * const t)
{
  if(Comp2RefEqual<int>::isInvalid(t))return -1;

  if(*t == *ref_)
    return 1;
  else
    return 0;
}

// run the test (result: [0, 1] or <0 for failure)
float Comp2RefEqualFloatROOT::runTest(const float * const t)
{
  if(Comp2RefEqual<float>::isInvalid(t))return -1;

  if(*t == *ref_)
    return 1;
  else
    return 0;
}

// true if test cannot run
bool Comp2RefEqualH1ROOT::isInvalid(const TH1F * const h)
{
  if(Comp2RefEqual<TH1F>::isInvalid(h))return true;
  
  // Check consistency of dimensions
  if (h->GetDimension() != 1 || ref_->GetDimension() != 1) {
    cerr << " Comp2RefEqualH1ROOT error: Histograms must be 1-D\n";
    return true;
  }
  
  TAxis *axis1 = h->GetXaxis();
  TAxis *axis2 = ref_->GetXaxis();
  ncx1   = axis1->GetNbins();
  ncx2   = axis2->GetNbins();
  // Check consistency in number of channels
  if (ncx1 != ncx2) {
    cerr << " Comp2RefEqualH1ROOT error: different number of channels! (" 
	 << ncx1 << ", " << ncx2 << ") " << endl;
    return true;
  }
  
  // if here, everything is good
  return false;
}

// run the test (result: [0, 1] or <0 for failure)
float Comp2RefEqualH1ROOT::runTest(const TH1F * const h)
{
  badChannels_.clear();
  if(isInvalid(h)) return -1;
  
  // use underflow bin
  Int_t first = 0; // 1 
  // use overflow bin
  Int_t last  = ncx1+1; // ncx1

  bool failure = false;
  for (Int_t bin=first;bin<=last;bin++) 
    {
      float contents = h->GetBinContent(bin);
      if (contents != ref_->GetBinContent(bin))
	{
	  failure = true;
	  dqm::me_util::Channel chan(bin, 0, 0, contents, 
				     h->GetBinError(bin));
	  badChannels_.push_back(chan);
	}
    }

  if(failure)
    return 0;
  else
    return 1;
}

// true if test cannot run
bool Comp2RefEqualH2ROOT::isInvalid(const TH2F * const h)
{
  if(Comp2RefEqual<TH2F>::isInvalid(h))return true;
  
  // Check consistency of dimensions
  if (h->GetDimension() != 2 || ref_->GetDimension() != 2) {
    cerr << " Comp2RefEqualH2ROOT error: Histograms must be 2-D\n";
    return true;
  }
  
  TAxis *xaxis1 = h->GetXaxis();
  TAxis *xaxis2 = ref_->GetXaxis();
  TAxis *yaxis1 = h->GetYaxis();
  TAxis *yaxis2 = ref_->GetYaxis();
  ncx1   = xaxis1->GetNbins();
  ncx2   = xaxis2->GetNbins();
  ncy1   = yaxis1->GetNbins();
  ncy2   = yaxis2->GetNbins();
  
  // Check consistency in number of X-channels
  if (ncx1 != ncx2) {
    cerr << " Comp2RefEqualH2ROOT error: different number of X-channels! (" 
	 << ncx1 << ", " << ncx2 << ") " << endl;
    return true;
  }
  
  // Check consistency in number of Y-channels
  if (ncy1 != ncy2) {
    cerr << " Comp2RefEqualH2ROOT error: different number of Y-channels! (" 
	 << ncy1 << ", " << ncy2 << ") " << endl;
    return true;
  }
  
  // if here, everything is good
  return false;
}

// run the test (result: [0, 1] or <0 for failure)
float Comp2RefEqualH2ROOT::runTest(const TH2F * const h)
{
  badChannels_.clear();
  if(isInvalid(h)) return -1;
  
  // use underflow bin
  Int_t firstX = 0; // 1 
  // use overflow bin
  Int_t lastX  = ncx1+1; // ncx1
  // use underflow bin
  Int_t firstY = 0; // 1 
  // use overflow bin
  Int_t lastY  = ncy1+1; // ncy1
  
  bool failure = false;
  Int_t binx, biny;
  for(biny = firstY; biny <= lastY; ++biny){
    for(binx = firstX; binx <= lastX; ++binx){
      float contents = h->GetBinContent(binx, biny);
      if(ref_->GetBinContent(binx,biny) != contents)
	{
	  failure = true;
	  dqm::me_util::Channel chan(binx, biny, 0, contents, 
				     h->GetBinError(binx, biny));
	  badChannels_.push_back(chan);
	}
    }
  }

  if(failure)
    return 0;
  else
    return 1;
}

// true if test cannot run
bool Comp2RefEqualH3ROOT::isInvalid(const TH3F * const h)
{
  if(Comp2RefEqual<TH3F>::isInvalid(h))return true;
  
  // Check consistency of dimensions
  if (h->GetDimension() != 3 || ref_->GetDimension() != 3) {
    cerr << " Comp2RefEqualH3ROOT error: Histograms must be 3-D\n";
    return true;
  }
  
  TAxis *xaxis1 = h->GetXaxis();
  TAxis *xaxis2 = ref_->GetXaxis();
  TAxis *yaxis1 = h->GetYaxis();
  TAxis *yaxis2 = ref_->GetYaxis();
  TAxis *zaxis1 = h->GetZaxis();
  TAxis *zaxis2 = ref_->GetZaxis();
  ncx1   = xaxis1->GetNbins();
  ncx2   = xaxis2->GetNbins();
  ncy1   = yaxis1->GetNbins();
  ncy2   = yaxis2->GetNbins();
  ncz1   = zaxis1->GetNbins();
  ncz2   = zaxis2->GetNbins();
    
  // Check consistency of dimensions
  if (h->GetDimension() != 3 || ref_->GetDimension() != 3) {
    cerr << " Comp2RefEqualH3ROOT error: Histograms must be 3-D\n";
    return false;
  }
  
  // Check consistency in number of X-channels
  if (ncx1 != ncx2) {
    cerr << " Comp2RefEqualH3ROOT error: different number of X-channels! (" 
	 << ncx1 << ", " << ncx2 << ") " << endl;
    return false;
  }
  
  // Check consistency in number of Y-channels
  if (ncy1 != ncy2) {
    cerr << " Comp2RefEqualH3ROOT error: different number of Y-channels! (" 
	 << ncy1 << ", " << ncy2 << ") " << endl;
    return false;
  }
  
  // Check consistency in number of Z-channels
  if (ncz1 != ncz2) {
    cerr << " Comp2RefEqualH3ROOT error: different number of Z-channels! (" 
	 << ncz1 << ", " << ncz2 << ") " << endl;
    return false;
    }
  
  // if here, everything is good
  return false;
}

// run the test (result: [0, 1] or <0 for failure)
float Comp2RefEqualH3ROOT::runTest(const TH3F * const h)
{
  badChannels_.clear();
  if(isInvalid(h)) return -1;

  // use underflow bin
  Int_t firstX = 0; // 1 
  // use overflow bin
  Int_t lastX  = ncx1+1; // ncx1
  // use underflow bin
  Int_t firstY = 0; // 1 
  // use overflow bin
  Int_t lastY  = ncy1+1; // ncy1
  // use underflow bin
  Int_t firstZ = 0; // 1 
  // use overflow bin
  Int_t lastZ  = ncz1+1; // ncz1

  bool failure = false;
  Int_t binx, biny, binz;
  for(binz = firstZ; binz <= lastZ; ++binz){
    for(biny = firstY; biny <= lastY; ++biny){
      for(binx = firstX; binx <= lastX; ++binx) {

	float contents = h->GetBinContent(binx, biny, binz);
	if(ref_->GetBinContent(binx,biny, binz) != contents)
	  {
	    failure = true;
	    dqm::me_util::Channel chan(binx, biny, binz, contents,
				       h->GetBinError(binx, biny, binz));
	    badChannels_.push_back(chan);
	  }
	
      }
    }
  }

  if(failure)
    return 0;
  else
    return 1;

}
