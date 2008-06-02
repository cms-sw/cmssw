#include "EventFilter/CSCRawToDigi/interface/CSCCLCTData.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <stdio.h>

bool CSCCLCTData::debug = false;


CSCCLCTData::CSCCLCTData(int ncfebs, int ntbins)
: ncfebs_(ncfebs), ntbins_(ntbins) 
{
  size_ = nlines();
  
  // initialize the words
  for(int ifeb = 0; ifeb < ncfebs_; ++ifeb) 
    {
      for(int tbin = 0; tbin < ntbins_; ++tbin)
	{
	  for(int layer = 1; layer <= 6; ++layer) 
	    {
	      dataWord(ifeb, tbin, layer) = CSCCLCTDataWord(ifeb, tbin, 0);
	    }
	}
    }
}
     


CSCCLCTData::CSCCLCTData(int ncfebs, int ntbins, const unsigned short * buf)
: ncfebs_(ncfebs), ntbins_(ntbins) 
{
  // add two more for odd ntbins, plus one for the e0c
  // Oct 2004 Rick: e0c line belongs to CSCTMBTrailer
  size_ = (nlines()%2==1)? nlines()+2 : nlines();
  
  memcpy(theData, buf, size_*2);
  
}

std::vector<CSCComparatorDigi>  CSCCLCTData::comparatorDigis(uint32_t idlayer, unsigned cfeb) 
{
  bool me1a = (CSCDetId::station(idlayer)==1) && (CSCDetId::ring(idlayer)==4);
  bool zplus = (CSCDetId::endcap(idlayer) == 1); 
  bool me1b = (CSCDetId::station(idlayer)==1) && (CSCDetId::ring(idlayer)==1);
  unsigned layer = CSCDetId::layer(idlayer);

  //looking for comp output on layer
  std::vector<CSCComparatorDigi> result;
  assert(layer>0 && layer<= 6);
  // this is pretty sparse data, so I wish we could check the
  // data word by word, not bit by bit, but I don't see how to
  // do the time sequencing that way.
  for(int distrip = 0; distrip < 8; ++distrip) 
    {
      uint16_t tbinbitsS0HS0=0;
      uint16_t tbinbitsS0HS1=0;
      uint16_t tbinbitsS1HS0=0;
      uint16_t tbinbitsS1HS1=0;
      for(int tbin = 0; tbin < ntbins_-2; ++tbin) 
	{
	  if(bitValue(cfeb, tbin, layer, distrip))
	    {
	      /// first do some checks
	      CSCCLCTDataWord word = dataWord(cfeb, tbin, layer);
	      assert(word.tbin_ == tbin);
	      assert(word.cfeb_ == cfeb);
	      // we have a hit.  The next two time samples
	      // are the other two bits in the triad
	      int bit2 = bitValue(cfeb, tbin+1, layer, distrip);
	      int bit3 = bitValue(cfeb, tbin+2, layer, distrip);
	      // should count from zero
	      int chamberDistrip = distrip + cfeb*8;
	      int HalfStrip = 4*chamberDistrip + bit2*2 + bit3;
	      int output = 4 + bit2*2 + bit3;
	      /*
	       * Handles distrip logic; comparator output is for pairs of strips:
	       * hit  bin  dec
	       * x--- 100   4
	       * -x-- 101   5
	       * --x- 110   6
	       * ---x 111   7
	       *
	       */

	      if (debug)
		edm::LogInfo ("CSCCLCTData")
		  << "fillComparatorOutputs: layer = "
		  << layer << " timebin = " << tbin
		  << " cfeb = " << cfeb << " distrip = " << chamberDistrip
		  << " HalfStrip = " << HalfStrip
		  << " Output " << output;

	      ///what is actually stored in comparator digis are 0/1 for left/right halfstrip for each strip

	      ///constructing four bitted words for tbits on
	      if (output==4) tbinbitsS0HS0=tbinbitsS0HS0+(1<<tbin);      
	      if (output==5) tbinbitsS0HS1=tbinbitsS0HS1+(1<<tbin);
	      if (output==6) tbinbitsS1HS0=tbinbitsS1HS0+(1<<tbin);
	      if (output==7) tbinbitsS1HS1=tbinbitsS1HS1+(1<<tbin);
	      
	      tbin += 2;
	    }
	}//end of loop over time bins
      //we do not have to check over the last couple of time bins if there are no hits since
      //comparators take 3 time bins

      // Store digis each of possible four halfstrips for given distrip:
      if (tbinbitsS0HS0 || tbinbitsS0HS1 || tbinbitsS1HS0 || tbinbitsS1HS1) {
	unsigned int cfeb_corr    = cfeb;
	unsigned int distrip_corr = distrip;

	// Fix ordering of strips and CFEBs in ME1/1.
	// SV, 27/05/08: keep CFEB=4 for ME1/a until CLCT trigger logic
	// stops combining it with the info from the other 4 CFEBs (ME1/b).
	// if ( me1a )           { cfeb_corr = 0; } // reset 4 to 0
	if ( me1a &&  zplus ) { distrip_corr = 7-distrip; } // 0-7 -> 7-0
	if ( me1b && !zplus ) { distrip_corr = 7-distrip; cfeb_corr = 3-cfeb; }

	int strip = 16*cfeb_corr + 2*distrip_corr + 1;

	if (debug)
	  edm::LogInfo ("CSCCLCTData")
	    << "fillComparatorOutputs: cfeb_corr = " << cfeb_corr
	    << " distrip_corr = " << distrip_corr << " strip = " << strip;

	if (( me1a && zplus ) || ( me1b && !zplus )) {
	  // Half-strips need to be flipped too.
	  if (tbinbitsS1HS1) result.push_back(CSCComparatorDigi(strip, 0, tbinbitsS1HS1));
	  if (tbinbitsS1HS0) result.push_back(CSCComparatorDigi(strip, 1, tbinbitsS1HS0));
	  if (tbinbitsS0HS1) result.push_back(CSCComparatorDigi(strip+1, 0, tbinbitsS0HS1));
	  if (tbinbitsS0HS0) result.push_back(CSCComparatorDigi(strip+1, 1, tbinbitsS0HS0));
	}
	else {
	  if (tbinbitsS0HS0) result.push_back(CSCComparatorDigi(strip, 0, tbinbitsS0HS0));
	  if (tbinbitsS0HS1) result.push_back(CSCComparatorDigi(strip, 1, tbinbitsS0HS1));
	  if (tbinbitsS1HS0) result.push_back(CSCComparatorDigi(strip+1, 0, tbinbitsS1HS0));
	  if (tbinbitsS1HS1) result.push_back(CSCComparatorDigi(strip+1, 1, tbinbitsS1HS1));
	}
	//uh oh ugly ugly ugly!
      }
    }//end of loop over distrips
  
  return result;
}



std::vector<CSCComparatorDigi>  CSCCLCTData::comparatorDigis(int layer) 
{
  //returns comparators for one layer for all cfebs
  std::vector<CSCComparatorDigi> result;
  assert(layer>0 && layer<= 6);

  for(int cfeb = 0; cfeb < ncfebs_; ++cfeb) 
    {
      std::vector<CSCComparatorDigi> oneCfebDigi = comparatorDigis(layer,cfeb);
      result.insert(result.end(), oneCfebDigi.begin(), oneCfebDigi.end());      
    }

  return result;
}


bool CSCCLCTData::check() const 
{
  bool result = true;
  for(int cfeb = 0; cfeb < ncfebs_; ++cfeb) 
    {
      for(int tbin = 0; tbin < ntbins_; ++tbin)
	{
	  for(int layer = 1; layer <= 6; ++layer) 
	    {
	      /// first do some checks
	      const CSCCLCTDataWord & word = dataWord(cfeb, tbin, layer);
	      bool wordIsGood = (word.tbin_ == tbin) && (word.cfeb_ == cfeb);
	      result = result && wordIsGood;
	      if(!wordIsGood && debug)
		{
		  edm::LogError("CSCCLCTData") << "Bad CLCT data  in layer " << layer 
					       << " expect CFEB " << cfeb << " tbin " << tbin;
		  edm::LogError("CSCCLCTData") << " See " << word.cfeb_ << " " 
					       << word.tbin_;
		}
	    }
	}
    }
  if(!result) edm::LogError("CSCCLCTData") << "++ Bad CLCT Data ++ ";
  return result;
}

