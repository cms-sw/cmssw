
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBTimeSlice.h"
#include "EventFilter/CSCRawToDigi/interface/CSCBadCFEBTimeSlice.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCFEBStatusDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cassert>

CSCCFEBData::CSCCFEBData(unsigned number, unsigned short * buf, uint16_t format_version, bool f_dcfeb) 
  : theSize(0), boardNumber_(number), theNumberOfSamples(0), theFormatVersion(format_version), fDCFEB(f_dcfeb) {
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
      //store bad word for status digis
      bWords.push_back(badSlice->word(1).data()); //all 4 words are assumed identical so saving #1 only  
    } 
    else {
      // OK.  Maybe it's good.
      CSCCFEBTimeSlice * goodSlice 
	= reinterpret_cast<CSCCFEBTimeSlice *>(buf+pos);
      if(goodSlice->check()) {
	// show that a good slice starts here
	theSliceStarts.push_back(std::pair<int, bool>(pos, true));
	// it will just be an array of CSCCFEBTimeSlices, so we'll
	// grab the number of time slices from the first good one
	// !!! VB - Limit maximum number of CFEB samples to 8. 
	// !!!      In Run2 rare CFEB data corruptions were causing RECO problems with mistakenly setting 16 samples flags
	// !!!      Will need another fix in case of CSC switch to 16 samples readout
	// maxSamples =   goodSlice->sixteenSamples() ? 16 : 8;
	if (goodSlice->sixteenSamples()) LogTrace ("CSCCFEBData|CSCRawToDigi")
          << "CFEB DATA slice " << theNumberOfSamples << " 16 samples flag is detected";
	pos += goodSlice->sizeInWords();
      } 
      else {
	LogTrace ("CSCCFEBData|CSCRawToDigi") 
	  << "CORRUPT CFEB DATA slice " << theNumberOfSamples << std::hex << " " 
	  << *(buf+pos+3) << " " << *(buf+pos+2) << " "  << *(buf+pos+1) << " "<< *(buf+pos);
	//ok slice is bad but try another one at 100 words after it
        theSliceStarts.push_back(std::pair<int, bool>(pos, false));
	pos += 100;
      }
    }
    ++theNumberOfSamples;
  }
  theSize = pos;
  memcpy(theData, buf, theSize*2);
}


CSCCFEBData::CSCCFEBData(unsigned number, bool sixteenSamples, uint16_t format_version, bool f_dcfeb) 
: boardNumber_(number), theNumberOfSamples(sixteenSamples ? 16 : 8), theFormatVersion(format_version), fDCFEB(f_dcfeb)
{
  theSliceStarts.reserve(theNumberOfSamples);

  // fill the SCA controller words
  CSCCFEBSCAControllerWord scaWord;
  scaWord.ts_flag = sixteenSamples;

  // make a template slice to copy into theData buffer
  CSCCFEBTimeSlice slice;
  slice.setControllerWord(scaWord);

  for(unsigned i = 0; i < theNumberOfSamples; ++i) 
    {
      unsigned short * pos = theData+i*100;
      memcpy(pos, &slice, 200);
      theSliceStarts.push_back(std::pair<int,bool>(i*100, true));
    }
  theSize = theNumberOfSamples*100;
}

void CSCCFEBData::add(const CSCStripDigi & digi, int layer)
{
  std::vector<int> scaCounts =  digi.getADCCounts();
  for(unsigned itime = 0; itime < theNumberOfSamples; ++itime) 
    {
      unsigned channel = (digi.getStrip()-1) % 16 + 1;
      unsigned value = scaCounts[itime] & 0xFFF; // 12-bit
      // assume it's good, since we're working with simulation
      const CSCCFEBTimeSlice * slice = timeSlice(itime);
      assert(slice != 0);
      slice->timeSample(layer, channel,fDCFEB)->adcCounts = value;
      /// =VB= Set CRC value for simulated data
      ((CSCCFEBTimeSlice *)slice)->setCRC();
    }
}

const CSCCFEBTimeSlice * CSCCFEBData::timeSlice(unsigned i) const 
{
  const CSCCFEBTimeSlice * result;
  assert(i < theNumberOfSamples);
  std::pair<int,bool> start = theSliceStarts[i];
  // give a NULL pointer if this is a bad slice
  if(!start.second) 
    {
      result = 0;
    } 
  else 
    {
      result = reinterpret_cast<const CSCCFEBTimeSlice *>(theData+start.first);
    }
  return result;
}


unsigned CSCCFEBData::adcCounts(unsigned layer, unsigned channel, unsigned timeBin) const 
{
  unsigned result = 0;
  const CSCCFEBTimeSlice * slice = timeSlice(timeBin);
  // zero is returned for bad slices
  if(slice) result = slice->timeSample(layer, channel,fDCFEB)->adcCounts;
  return result;
}
unsigned CSCCFEBData::adcOverflow(unsigned layer, unsigned channel, unsigned timeBin) const 
{
  unsigned result = 0;
  const CSCCFEBTimeSlice * slice = timeSlice(timeBin);
  // zero is returned for bad slices
  if(slice) result = slice->timeSample(layer, channel,fDCFEB)->adcOverflow;
  return result;
}

unsigned CSCCFEBData::controllerData(unsigned uglay, unsigned ugchan, unsigned timeBin) const 
{

// The argument notation is
// uglay = un-Gray Coded layer index 1-6
// ugchan = un-Gray Coded channel index 1-16
// The point being that the SCAC is serially encoded directly in the data stream (without Gray Coding)
// so the layer and channel indexes here are just the direct ordering into the data stream.

  unsigned result = 0;
  const CSCCFEBTimeSlice * slice = timeSlice(timeBin);
  // zero is returned for bad slices
  if(slice) result = slice->timeSample( (ugchan-1)*6+uglay-1 )->controllerData;
  return result;
}

unsigned CSCCFEBData::overlappedSampleFlag(unsigned layer, unsigned channel, unsigned timeBin) const 
{
  unsigned result = 0;
  const CSCCFEBTimeSlice * slice = timeSlice(timeBin);
  // zero is returned for bad slices
  if(slice) result = slice->timeSample(layer, channel,fDCFEB)->overlappedSampleFlag;
  return result;
}
unsigned CSCCFEBData::errorstat(unsigned layer, unsigned channel, unsigned timeBin) const 
{
  unsigned result = 0;
  const CSCCFEBTimeSlice * slice = timeSlice(timeBin);
  // zero is returned for bad slices
  if(slice) result = slice->timeSample(layer, channel,fDCFEB)->errorstat;
  return result;
}


void CSCCFEBData::setL1A(unsigned l1a)
{
  for (unsigned i=0; i < theNumberOfSamples; i++) setL1A(i, l1a); 
}

void CSCCFEBData::setL1A(unsigned i, unsigned l1a)
{
  assert(i < theNumberOfSamples);
  std::pair<int,bool> start = theSliceStarts[i];
  // give a NULL pointer if this is a bad slice
  if(start.second)
    {
      (reinterpret_cast<CSCCFEBTimeSlice *>(theData+start.first))->set_L1Anumber(l1a);
    }
}

CSCCFEBStatusDigi CSCCFEBData::statusDigi() const 
{
  ///returns one status digi per cfeb 
  ///contains bWord if slice is bad 
  ///also contains crc word and controller word

  std::vector<uint16_t> crcWords(nTimeSamples());
  std::vector<uint16_t> contrWords(nTimeSamples());

  if (nTimeSamples()==0) 
    {
      LogTrace("CSCCFEBData|CSCRawToDigi") << "nTimeSamples is zero - CFEB data corrupt?";
    }
  else
    {
      for(unsigned itime = 0; itime < nTimeSamples(); ++itime) {
	const CSCCFEBTimeSlice * slice = timeSlice(itime);
	// zero is returned for bad slices
	if (slice) crcWords[itime] = slice->get_crc();
	if (slice) 
	  {	
	    int layer=1; ///here layer=1 bec this word repeats 6 times for each layer
	    for(unsigned i = 0; i < 16; ++i)
	      {
		contrWords[itime] |= slice->timeSample(i*6+layer-1)->controllerData << i;
	      }
	  }

      }
    }

  CSCCFEBStatusDigi result(boardNumber_+1, crcWords, contrWords, bWords);
  return result;
}



void CSCCFEBData::digis(uint32_t idlayer, std::vector<CSCStripDigi> & result )
{
  
  // assert(layer>0 && layer <= 6);

  LogTrace("CSCCFEBData|CSCRawToDigi") << "nTimeSamples in CSCCFEBData::digis = " << nTimeSamples();
  if (nTimeSamples()==0) {
     LogTrace("CSCCFEBData|CSCRawToDigi") << "nTimeSamples is zero - CFEB data corrupt?";
     return;
  }

  result.reserve(16);

  std::vector<int> sca(nTimeSamples());
  std::vector<uint16_t> overflow(nTimeSamples());
  std::vector<uint16_t> overlap(nTimeSamples());
  std::vector<uint16_t> errorfl(nTimeSamples());

  bool me1a = (CSCDetId::station(idlayer)==1) && (CSCDetId::ring(idlayer)==4);
  bool zplus = (CSCDetId::endcap(idlayer) == 1); 
  bool me1b = (CSCDetId::station(idlayer)==1) && (CSCDetId::ring(idlayer)==1);
  
  unsigned layer = CSCDetId::layer(idlayer);

  std::vector<uint16_t> l1a_phase(nTimeSamples());
  for(unsigned itime = 0; itime < nTimeSamples(); ++itime) {
    l1a_phase[itime] = controllerData(layer, 13, itime); // will be zero if timeslice bad
    LogTrace("CSCCFEBData|CSCRawToDigi") << CSCDetId(idlayer) << " time sample " << itime+1 << " l1a_phase = " << controllerData(layer, 13, itime);
    LogTrace("CSCCFEBData|CSCRawToDigi") << CSCDetId(idlayer) << " time sample " << itime+1 << " lct_phase = " << controllerData(layer, 14, itime);
    LogTrace("CSCCFEBData|CSCRawToDigi") << CSCDetId(idlayer) << " time sample " << itime+1 << " # samples = " << controllerData(layer, 16, itime);
  };

  for(unsigned ichannel = 1; ichannel <= 16; ++ichannel)
    {
      // What is the point of testing here? Move it outside this loop
      //      if (nTimeSamples()==0)
      //	{
      //	  LogTrace("CSCCFEBData|CSCRawToDigi") << "nTimeSamples is zero - CFEB data corrupt?";
      //	  break;
      //	}
      
      for(unsigned itime = 0; itime < nTimeSamples(); ++itime)
	{
	  const CSCCFEBTimeSlice * slice = timeSlice(itime);
	  if (slice)
	    {
	      CSCCFEBDataWord * word;
	      word = slice->timeSample(layer, ichannel,fDCFEB);
	      if (word)
		{  ///for bad or missing data word will be zero
		  sca[itime] = word->adcCounts;
		  overflow[itime] = word->adcOverflow;
		  overlap[itime] = word->overlappedSampleFlag;
		  errorfl[itime] = word->errorstat;

		  // Stick the l1a_phase bit into 'overlap' too (so we can store it in CSCStripDigi
		  // without changing CSCStripDigi format). 
		  // Put it in the 9th bit of the overlap word which is only 1-bit anyway.
                  overlap[itime] = (( l1a_phase[itime] & 0x1 ) << 8 ) | ( word->overlappedSampleFlag & 0x1 );
		}
	    }
	}
      if (sca.empty())
	{
	  LogTrace("CSCCFEBData|CSCRawToDigi") << "ADC counts empty - CFEB data corrupt?";
	  break;
	}
      int strip = ichannel + 16*boardNumber_;

      if (theFormatVersion == 2013) { /// Handle 2013 Format 

         if ( me1a ) strip = strip%64; // reset 65-112/ to 1-48 digi
         if ( me1a && zplus ) { strip = 49 - strip; } // 1-48 -> 48-1 
         if ( me1b && !zplus) { strip = 65 - strip;} // 1-64 -> 64-1 ...

      } else { // Handle original 2005 format
     
         if ( me1a ) strip = strip%64; // reset 65-80 to 1-16 digi
         if ( me1a && zplus ) { strip = 17 - strip; } // 1-16 -> 16-1 
         if ( me1b && !zplus) { strip = 65 - strip;} // 1-64 -> 64-1 ...
      }
      result.push_back(CSCStripDigi(strip, sca, overflow, overlap, errorfl));
    } 
}



std::vector<CSCStripDigi> CSCCFEBData::digis(unsigned idlayer)
{
  //assert(layer>0 && layer <= 6);
  std::vector<CSCStripDigi> result;
  uint32_t layer= idlayer;
  digis(layer, result);
  return result;
}



bool CSCCFEBData::check() const 
{
  bool result = true;
  for(unsigned i = 0; i < theNumberOfSamples; ++i) 
    {
      const CSCCFEBTimeSlice * slice = timeSlice(i);
      if(slice==0 || !timeSlice(i)->check()) result = false;
    }
  return result;
}

std::ostream & operator<<(std::ostream & os, const CSCCFEBData & data) 
{
  os << "printing CFEB data sample by sample " << std::endl;
  for(unsigned ilayer = 1; ilayer <= 6; ++ilayer) 
    {
      for(unsigned channel = 1; channel <= 16; ++channel) 
	{
	  unsigned strip = channel + data.boardNumber_*16;
	  os << "Strip " << strip << " ";
	  for(unsigned timeBin = 0; timeBin < data.nTimeSamples(); ++timeBin)
	    {
	      os << data.adcCounts(ilayer, channel, timeBin) << " " ;
	    }
	  os << std::endl;
	}
    }
  return os;
}

std::vector < std::vector<CSCStripDigi> > CSCCFEBData::stripDigis()
{
  std::vector < std::vector<CSCStripDigi> > result;
  for (int layer = 1; layer <= 6; ++layer) 
    {
      result.push_back(digis(layer));
    }
  return result;
}

