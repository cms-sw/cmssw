#include "DQMServices/Core/interface/ContentsWithinRangeROOT.h"

#include <iostream>

using dqm::me_util::Channel;

using std::cerr; using std::endl;

// run the test (result: fraction of entries [*not* bins!] within X-range)
// [0, 1] or <0 for failure
float ContentsXRangeROOT::runTest(const TH1F * const h)
{
  if(!h)
    return -1;

  TAxis * axis = h->GetXaxis();
  if(!rangeInitialized)
    {
      if(axis)
	setAllowedXRange(axis->GetXmin(), axis->GetXmax());
      else
	return -1;
    }
  Int_t ncx = axis->GetNbins();
  // use underflow bin
  Int_t first = 0; // 1 
  // use overflow bin
  Int_t last  = ncx+1; // ncx
  // all entries
  Double_t sum = 0;
  // entries outside X-range
  Double_t fail = 0;
  Int_t bin;
  for (bin = first; bin <= last; ++bin) 
    {
      Double_t contents = h->GetBinContent(bin);
      float x = h->GetBinCenter(bin);
      sum += contents;
      if(x < xmin_ || x > xmax_)fail += contents;
    }
  
  // return fraction of entries within allowed X-range
  return (sum - fail)/sum;
}

// run the test (result: fraction of bins [*not* entries!] that passed test)
// [0, 1] or <0 for failure
float ContentsYRangeROOT::runTest(const TH1F * const h)
{
  badChannels_.clear();

  if(!h)
    return -1;
  
  TAxis * axis = h->GetXaxis();
  if(!rangeInitialized || !axis)
    return 1; // all bins are accepted if no initialization

  Int_t ncx = axis->GetNbins();
  // do NOT use underflow bin
  Int_t first = 1; 
  // do NOT use overflow bin
  Int_t last  = ncx;
  // bins outside Y-range
  Int_t fail = 0;
  Int_t bin;
  for (bin = first; bin <= last; ++bin) 
    {
      Double_t contents = h->GetBinContent(bin);
      bool failure = false;
      if(deadChanAlgo_)
	// dead channel: equal to or less than ymin_
	failure = contents <= ymin_;
      else
	// allowed y-range: [ymin_, ymax_]
	failure = (contents < ymin_ || contents > ymax_);
      
      if(failure)
	{
	  Channel chan(bin, 0, 0, contents, h->GetBinError(bin));
	  badChannels_.push_back(chan);
	  ++fail;
	}
    }
  
  // return fraction of bins that passed test
  return 1.*(ncx - fail)/ncx;
}


// run the test (result: fraction of channels not appearing noisy or "hot")
// [0, 1] or <0 for failure
float NoisyChannelROOT::runTest(const TH1F * const h)
{
  badChannels_.clear();
  if(!h) return -1;
  
  TAxis * axis = h->GetXaxis();
  if(!rangeInitialized || !axis)
    return 1; // all channels are accepted if tolerance has not been set

  Int_t ncx = axis->GetNbins();
  // do NOT use underflow bin
  Int_t first = 1; 
  // do NOT use overflow bin
  Int_t last  = ncx;
  // bins outside Y-range
  Int_t fail = 0;
  Int_t bin;
  for (bin = first; bin <= last; ++bin) 
    {
      Double_t contents = h->GetBinContent(bin);
      Double_t average = getAverage(bin, h);
      bool failure = false;
      if(average != 0)
	failure = ( ((contents-average)/TMath::Abs(average)) > tolerance);
      
      if(failure)
	{
	  ++fail;
	  Channel chan(bin, 0, 0, contents, h->GetBinError(bin));
	  badChannels_.push_back(chan);
	}
    }
  
  // return fraction of bins that passed test
  return 1.*(ncx - fail)/ncx;
}

// get average for bin under consideration
// (see description of method setNumNeighbors)
Double_t NoisyChannelROOT::getAverage(int bin, const TH1F * const h) const
{
  // do NOT use underflow bin
  Int_t first = 1; 
  // do NOT use overflow bin
  Int_t ncx  = h->GetXaxis()->GetNbins();

  Double_t sum = 0; int bin_low, bin_hi;
  for(unsigned i = 1; i <= numNeighbors; ++i)
    {
      // use symmetric-to-bin bins to calculate average
      bin_low = bin-i; bin_hi = bin+i;
      // check if need to consider bins on other side of spectrum
      // (ie. if bins below 1 or above ncx)

      while(bin_low < first) // shift bin by +ncx
	{bin_low = ncx + bin_low;}
      while(bin_hi > ncx) // shift bin by -ncx
	{bin_hi = bin_hi - ncx;}

      sum += h->GetBinContent(bin_low) + h->GetBinContent(bin_hi);
    }
  
  // average is sum over the # of bins used
  return sum/(numNeighbors * 2);
}


// run the test (result: fraction of channels that passed test);
// [0, 1] or <0 for failure
// implementation: Giuseppe Della Ricca
float ContentsTH2FWithinRangeROOT::runTest(const TH2F * const h)
{

  badChannels_.clear();

  if ( !h ) return -1;
  if ( isInvalid() )return -1;

  TAxis* xaxis = h->GetXaxis();
  if ( !xaxis ) return -1;

  TAxis* yaxis = h->GetYaxis();
  if ( !yaxis ) return -1;

  int ncx = xaxis->GetNbins();
  int ncy = yaxis->GetNbins();

  int fail = 0;

  int nsum = 0;
  float sum = 0.0;
  
  float average = 0.0;

  if (checkMeanTolerance_) 
    { // calculate average value of all bin contents

      for (int cx = 1; cx <= ncx; ++cx ) {
	for (int cy = 1; cy <= ncy; ++cy ) {

	  sum += h->GetBinContent(h->GetBin(cx, cy));
	  ++nsum;
	}
      }

      if (nsum > 0) average = sum/nsum;
      
    } // calculate average value of all bin contents
  
  for ( int cx = 1; cx <= ncx; ++cx ) {
    for ( int cy = 1; cy <= ncy; ++cy ) {

      bool failMean = false;
      bool failRMS = false;
      bool failMeanTolerance = false;

      if ( checkMean_ ) {
        float mean = h->GetBinContent(h->GetBin(cx, cy));
        failMean = (mean < minMean_ || mean > maxMean_);
      }

      if ( checkRMS_ ) {
        float rms = h->GetBinError(h->GetBin(cx, cy));
        failRMS = (rms < minRMS_ || rms > maxRMS_);
      }

      if ( checkMeanTolerance_ ) {
        float mean = h->GetBinContent(h->GetBin(cx, cy));
        failMeanTolerance = ( TMath::Abs(mean - average) > 
			      toleranceMean_ * TMath::Abs(average) );
      }

      if ( failMean || failRMS || failMeanTolerance ) {

        Channel chan(cx, cy, 0, h->GetBinContent(h->GetBin(cx, cy)),
		     h->GetBinError(h->GetBin(cx, cy)) );
        badChannels_.push_back(chan);
        ++fail;
      }

    }
  }

  return 1.*(ncx*ncy - fail)/(ncx*ncy);

}

// check that allowed range is logical
void ContentsTH2FWithinRangeROOT::checkRange(const float xmin, const float xmax)
{

  if( xmin < xmax ) 
    {
      validMethod_ = true;
    } 
  else 
    {
      cerr << " *** Error! Illogical range: (" << xmin << ", " << xmax 
	   << ") in algorithm " << getAlgoName() << endl;
      validMethod_ = false;
    }
}

// run the test (result: fraction of channels that passed test);
// [0, 1] or <0 for failure
// implementation: Giuseppe Della Ricca
float ContentsProfWithinRangeROOT::runTest(const TProfile * const h)
{

  badChannels_.clear();

  if ( !h ) return -1;
  if ( isInvalid() )return -1;

  TAxis* xaxis = h->GetXaxis();
  if ( !xaxis ) return -1;

  int ncx = xaxis->GetNbins();

  int fail = 0;

  int nsum = 0;
  float sum = 0.0;
  
  float average = 0.0;

  if (checkMeanTolerance_) 
    { // calculate average value of all bin contents
    
      for (int cx = 1; cx <= ncx; ++cx ) {

	if ( h->GetBinEntries(h->GetBin(cx)) >= min_entries_/(ncx) ) {
	  sum += h->GetBinContent(h->GetBin(cx));
	  ++nsum;
	}
      }

      if (nsum > 0) average = sum/nsum;

    } // calculate average value of all bin contents
  
  for (int cx = 1; cx <= ncx; ++cx ) 
    {
      
      bool failMean = false;
      bool failRMS = false;
      bool failMeanTolerance = false;

      if ( h->GetBinEntries(h->GetBin(cx)) >= min_entries_/(ncx) ) 
	{
	  if ( checkMean_ ) {
	    float mean = h->GetBinContent(h->GetBin(cx));
	    failMean = (mean < minMean_ || mean > maxMean_);
	  }
	  
	  if ( checkRMS_ ) {
	    float rms = h->GetBinError(h->GetBin(cx));
	    failRMS = (rms < minRMS_ || rms > maxRMS_);
	  }

          if ( checkMeanTolerance_ ) {
            float mean = h->GetBinContent(h->GetBin(cx));
            failMeanTolerance = ( TMath::Abs(mean - average) > 
				  toleranceMean_ * TMath::Abs(average) );
          }
	
	}

      if ( failMean || failRMS || failMeanTolerance ) 
	{
	  Channel chan(cx, int(h->GetBinEntries(h->GetBin(cx))), 
		       0, h->GetBinContent(h->GetBin(cx)),
		       h->GetBinError(h->GetBin(cx)) );
	  badChannels_.push_back(chan);
	  ++fail;
	}

  }

  return 1.*(ncx - fail)/(ncx);

}

// check that allowed range is logical
void ContentsProfWithinRangeROOT::checkRange(const float xmin, const float xmax)
{

  if( xmin < xmax ) 
    {
      validMethod_ = true;
    } 
  else 
    {
      cerr << " *** Error! Illogical range: (" << xmin << ", " << xmax 
	   << ") in algorithm " << getAlgoName() << endl;
      validMethod_ = false;
    }
}


// run the test (result: fraction of channels that passed test);
// [0, 1] or <0 for failure
// implementation: Giuseppe Della Ricca
float ContentsProf2DWithinRangeROOT::runTest(const TProfile2D * const h)
{

  badChannels_.clear();

  if ( !h ) return -1;
  if ( isInvalid() )return -1;

  TAxis* xaxis = h->GetXaxis();
  if ( !xaxis ) return -1;

  TAxis* yaxis = h->GetYaxis();
  if ( !yaxis ) return -1;

  int ncx = xaxis->GetNbins();
  int ncy = yaxis->GetNbins();

  int fail = 0;

  int nsum = 0;
  float sum = 0.0;

  float average = 0.0;

  if ( checkMeanTolerance_ ) 
    { // calculate average value of all bin contents

      for (int cx = 1; cx <= ncx; ++cx ) {
	for (int cy = 1; cy <= ncy; ++cy ) {
	  
	  if (h->GetBinEntries(h->GetBin(cx, cy)) >= 
	      min_entries_/(ncx*ncy) ) 
	    {
	      sum += h->GetBinContent(h->GetBin(cx, cy));
	      ++nsum;
	    }	  

	}
      }

      if (nsum > 0) average = sum/nsum;

    } // calculate average value of all bin contents
  
  for ( int cx = 1; cx <= ncx; ++cx ) {
    for ( int cy = 1; cy <= ncy; ++cy ) {

      bool failMean = false;
      bool failRMS = false;
      bool failMeanTolerance = false;

      if ( h->GetBinEntries(h->GetBin(cx, cy)) >= min_entries_/(ncx*ncy) )
	{
	  
	  if ( checkMean_ ) {
	    float mean = h->GetBinContent(h->GetBin(cx, cy));
	    failMean = (mean < minMean_ || mean > maxMean_);
	  }
	  
	  if ( checkRMS_ ) {
	    float rms = h->GetBinError(h->GetBin(cx, cy));
	    failRMS = (rms < minRMS_ || rms > maxRMS_);
	  }

          if ( checkMeanTolerance_ ) {
            float mean = h->GetBinContent(h->GetBin(cx, cy));
            failMeanTolerance = (TMath::Abs(mean - average) 
				 > toleranceMean_ * TMath::Abs(average) );
          }
	  
	}
      
      if ( failMean || failRMS || failMeanTolerance ) 
	{
	  Channel chan(cx, cy, int(h->GetBinEntries(h->GetBin(cx, cy))), 
		       h->GetBinContent(h->GetBin(cx, cy)),
		       h->GetBinError(h->GetBin(cx, cy)) );
	  badChannels_.push_back(chan);
	  ++fail;
	}

    }
  }

  return 1.*(ncx*ncy - fail)/(ncx*ncy);

}

// check that allowed range is logical
void ContentsProf2DWithinRangeROOT::checkRange(const float xmin, const float xmax)
{

  if( xmin < xmax ) 
    {
      validMethod_ = true;
    } 
  else 
    {
      cerr << " *** Error! Illogical range: (" << xmin << ", " << xmax 
	   << ") in algorithm " << getAlgoName() << endl;
      validMethod_ = false;
    }
}

