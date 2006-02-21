#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBTimeSlice.h"
#include "EventFilter/CSCRawToDigi/interface/CSCBadCFEBTimeSlice.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cassert>

CSCCFEBData::CSCCFEBData(unsigned number, unsigned short * buf) 
: theSize(0), boardNumber_(number), theNumberOfSamples(0) {
  // I may be grabbing too many words, but that's OK
  // parse for time slices
  unsigned pos = 0;
  // to be set later
  unsigned maxSamples = 8;
  theSliceStarts.reserve(8);
  while(theNumberOfSamples < maxSamples) {

    // first see if it's a bad slice
    CSCBadCFEBTimeSlice * badSlice
      = reinterpret_cast<CSCBadCFEBTimeSlice *>(buf+pos);
    if(badSlice->check()) {
      //show that a bad slice starts here
      theSliceStarts.push_back(std::pair<int, bool>(pos, false));
      pos += badSlice->sizeInWords();
    } else {
      // OK.  Maybe it's good.

      CSCCFEBTimeSlice * goodSlice 
       = reinterpret_cast<CSCCFEBTimeSlice *>(buf+pos);
      if(goodSlice->check()) {
        // show that a good slice starts here
        theSliceStarts.push_back(std::pair<int, bool>(pos, true));
        // it will just be an array of CSCCFEBTimeSlices, so we'll
        // grab the number of time slices from the first good one
        maxSamples =   goodSlice->sixteenSamples() ? 16 : 8;
        pos += goodSlice->sizeInWords();
      } else {
        edm::LogError ("CSCCFEBData") << "CORRUPT CFEB DATA slice " << theNumberOfSamples << std::ios::hex 
				      << " " << *(buf+pos) << " " << *(buf+pos+1) << " " 
				      << *(buf+pos+2) << std::ios::dec;
        return;
      }
    }
    ++theNumberOfSamples;
  }
  theSize = pos;
  memcpy(theData, buf, theSize*2);
}


CSCCFEBData::CSCCFEBData(unsigned number, bool sixteenSamples) 
: boardNumber_(number), theNumberOfSamples(sixteenSamples ? 16 : 8) {
  theSliceStarts.reserve(theNumberOfSamples);

  // fill the SCA controller words
  CSCCFEBSCAControllerWord scaWord;
  scaWord.ts_flag = sixteenSamples;

  // make a template slice to copy into theData buffer
  CSCCFEBTimeSlice slice;
  slice.setControllerWord(scaWord);

  for(unsigned i = 0; i < theNumberOfSamples; ++i) {
     unsigned short * pos = theData+i*100;
     memcpy(pos, &slice, 200);
     theSliceStarts.push_back(std::pair<int,bool>(i*100, true));
  }
  theSize = theNumberOfSamples*100;
}


const CSCCFEBTimeSlice * CSCCFEBData::timeSlice(unsigned i) const {
  const CSCCFEBTimeSlice * result;
  assert(i < theNumberOfSamples);
  std::pair<int,bool> start = theSliceStarts[i];
  // give a NULL pointer if this is a bad slice
  if(!start.second) {
    result = 0;
  } else {
    result = reinterpret_cast<const CSCCFEBTimeSlice *>(theData+start.first);
  }
  return result;
}


unsigned CSCCFEBData::adcCounts(unsigned layer, unsigned channel, unsigned timeBin) const {
  unsigned result = 0;
  const CSCCFEBTimeSlice * slice = timeSlice(timeBin);
  // zero is returned for bad slices
  if(slice) result = slice->timeSample(layer, channel)->adcCounts;
  return result;
}

std::vector<CSCStripDigi> CSCCFEBData::digis(unsigned layer) const {
  assert(layer>0 && layer <= 6);
  std::vector<CSCStripDigi> result;
  result.reserve(16);
  std::vector<int> sca(nTimeSamples());
  for(unsigned ichannel = 1; ichannel <= 16; ++ichannel) {
    for(unsigned itime = 0; itime < nTimeSamples(); ++itime) {
      sca[itime] = adcCounts(layer, ichannel, itime);
    }
   
    int strip = ichannel + 16*boardNumber_;
    CSCStripDigi digi(strip, sca);
    result.push_back(digi);
  }
  return result;
}


bool CSCCFEBData::check() const {
  bool result = true;
  for(unsigned i = 0; i < theNumberOfSamples; ++i) {
    const CSCCFEBTimeSlice * slice = timeSlice(i);
    if(slice==0 || !timeSlice(i)->check()) result = false;
  }
  return result;
}


std::ostream & operator<<(std::ostream & os, const CSCCFEBData & data) {
  os << "printing CFEB data sample by sample " << std::endl;
  for(unsigned ilayer = 1; ilayer <= 6; ++ilayer) {
    for(unsigned channel = 1; channel <= 16; ++channel) {
      unsigned strip = channel + data.boardNumber_*16;
      os << "Strip " << strip << " ";
      for(unsigned timeBin = 0; timeBin < data.nTimeSamples(); ++timeBin) {
        os << data.adcCounts(ilayer, channel, timeBin) << " " ;
      }
      os << std::endl;
    }
  }
  return os;
}

std::vector < std::vector<CSCStripDigi> > CSCCFEBData::stripDigis() const {
  std::vector < std::vector<CSCStripDigi> > result;
  for (int layer = 1; layer <= 6; ++layer) {
    result.push_back(digis(layer));
  }
  return result;
}

