#include "EventFilter/CSCRawToDigi/interface/CSCCLCTData.h"
#include <iostream>
#include <stdio.h>

bool CSCCLCTData::debug = false;


CSCCLCTData::CSCCLCTData(int ncfebs, int ntbins)
: ncfebs_(ncfebs), ntbins_(ntbins) {
  size_ = nlines();

  // initialize the words
  for(int ifeb = 0; ifeb < ncfebs_; ++ifeb) {
    for(int tbin = 0; tbin < ntbins_; ++tbin) {
      for(int layer = 1; layer <= 6; ++layer) {
        dataWord(ifeb, tbin, layer) = CSCCLCTDataWord(ifeb, tbin, 0);
      }
    }
  }
}
     


CSCCLCTData::CSCCLCTData(int ncfebs, int ntbins, const unsigned short * buf)
: ncfebs_(ncfebs), ntbins_(ntbins) 
{
  //printf("CFEBs %d NTBINs %d \n",ncfebs,ntbins);
  // add two more for odd ntbins, plus one for the e0c
  // Oct 2004 Rick: e0c line belongs to CSCTMBTrailer
  size_ = (nlines()%2==1)? nlines()+2 : nlines();
  if (debug) {
    //for (int loop=0; loop < size_; loop++ ) cout << "+++ CSCCLCTData  " << hex << buf[loop] << endl;
  }
  memcpy(theData, buf, size_*2);
  //
}

#ifndef UNPCK_ONLY

std::vector<CSCComparatorDigi>  CSCCLCTData::comparatorDigis(int layer) {
  //cout << "looking for comp output on layer " << layer << endl;
  std::vector<CSCComparatorDigi> result;
  assert(layer>0 && layer<= 6);
  // this is pretty sparse data, so I wish we could check the
  // data word by word, not bit by bit, but I don't see how to
  // do the time sequencing that way.
  for(int cfeb = 0; cfeb < ncfebs_; ++cfeb) {
    for(int distrip = 0; distrip < 8; ++distrip) {
      //Rick: why is this -2?
      for(int tbin = 0; tbin < ntbins_-2; ++tbin) {
        if(bitValue(cfeb, tbin, layer, distrip)) {
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
          if (debug)
	    std::cout << std::dec << "fillComparatorOutputs: layer = " 
		      << layer << " timebin = " << tbin
		      << " cfeb = " << cfeb
		      << " distrip = " << chamberDistrip
		      << " HalfStrip = " << HalfStrip 
		      << " Output " << output << std::endl;
          result.push_back(
			   CSCComparatorDigi(chamberDistrip, output)
			   );
          tbin += 2;
        }
      }
    }
  }
  return result;
}

/*
void CSCCLCTData::add(const CSCComparatorOutput & comparator, int layer) {
  int cfeb = comparator.element() / 8;
  int distrip = comparator.element() % 8;
  int tbin = (int) comparator.time() / 25;
  int output = comparator.output();
  // encode the comparator output in the next three bins for this distrip
  for(int bit = 0; bit < 3; ++bit) {
     dataWord(cfeb, tbin+2-bit, layer).set(distrip, (output>>bit)&1);
  }
}
*/
#endif


bool CSCCLCTData::check() const {
  bool result = true;
  for(int cfeb = 0; cfeb < ncfebs_; ++cfeb) {
    for(int tbin = 0; tbin < ntbins_; ++tbin) {
      for(int layer = 1; layer <= 6; ++layer) {
        /// first do some checks
        const CSCCLCTDataWord & word = dataWord(cfeb, tbin, layer);
        bool wordIsGood = (word.tbin_ == tbin) && (word.cfeb_ == cfeb);
        result = result && wordIsGood;
        if(!wordIsGood && debug) {
          std::cout << "Bad CLCT data  in layer " << layer 
		    << " expect CFEB " << cfeb << " tbin " << tbin;
          std::cout << " See " << word.cfeb_ << " " 
		    << word.tbin_ << std::endl;
        }
      }
    }
  }
  if(!result) std::cout << "++ Bad CLCT Data ++ " << std::endl;
  return result;
}

