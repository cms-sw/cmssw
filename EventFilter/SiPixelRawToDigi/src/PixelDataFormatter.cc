#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "CondFormats/SiPixelObjects/interface/PixelROC.h"


#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <bitset>
#include <sstream>
#include <iostream>

using namespace std;
using namespace edm;

const int PixelDataFormatter::LINK_bits = 6;
const int PixelDataFormatter::ROC_bits  = 5;
const int PixelDataFormatter::DCOL_bits = 5;
const int PixelDataFormatter::PXID_bits = 8;
const int PixelDataFormatter::ADC_bits  = 8;

const int PixelDataFormatter::ADC_shift  = 0;
const int PixelDataFormatter::PXID_shift = ADC_shift + ADC_bits;
const int PixelDataFormatter::DCOL_shift = PXID_shift + PXID_bits;
const int PixelDataFormatter::ROC_shift  = DCOL_shift + DCOL_bits;
const int PixelDataFormatter::LINK_shift = ROC_shift + ROC_bits;


PixelDataFormatter::PixelDataFormatter( const SiPixelFedCablingMap * map)
  : theDigiCounter(0), theWordCounter(0), theCablingMap(map)
{
  int s32 = sizeof(Word32);
  int s64 = sizeof(Word64);
  int s8  = sizeof(char);
  if ( s8 != 1 || s32 != 4*s8 || s64 != 2*s32) {
     LogError("**PixelDataFormatter**")
          <<" unexpected sizes: "
          <<"  size of char is: " << s8
          <<", size of Word32 is: " << s32
          <<", size of Word64 is: " << s64
          <<", send exception" ;
  }
}

void PixelDataFormatter::interpretRawData(int fedId, const FEDRawData& rawData, Digis& digis, bool includeErrors, Errors& errors)
{
    int nWords = rawData.size()/sizeof(Word64);
    if (nWords==0) return;

  SiPixelFrameConverter * converter = (theCablingMap) ? 
      new SiPixelFrameConverter(theCablingMap, fedId) : 0; 

    // check headers
    const Word64* header = reinterpret_cast<const Word64* >(rawData.data()); header--;
    bool moreHeaders = true;
    while (moreHeaders) {
      header++;
      FEDHeader fedHeader( reinterpret_cast<const unsigned char*>(header));
      LogTrace("")<<"HEADER:  " <<  print(*header);
      if ( !fedHeader.check() ) break; // throw exception?
      if ( fedHeader.sourceID() != fedId) { 
        LogError("PixelDataFormatter::interpretRawData, fedHeader.sourceID() != fedId")
              <<", sourceID = " <<fedHeader.sourceID()
              <<", fedId = "<<fedId<<", errorType = 32"; 
        if(includeErrors) {
	  int errorType = 32;
	  SiPixelRawDataError error(*header, errorType);
	  errors.push_back(error);
	}
      moreHeaders = fedHeader.moreHeaders();
      }
    }

    // check trailers
    const Word64* trailer = reinterpret_cast<const Word64* >(rawData.data())+(nWords-1); trailer++;
    bool moreTrailers = true;
    while (moreTrailers) {
      trailer--;
      FEDTrailer fedTrailer(reinterpret_cast<const unsigned char*>(trailer)); 
      LogTrace("")<<"TRAILER: " <<  print(*trailer);
      if ( !fedTrailer.check()) { 
	if(includeErrors) {
	  int errorType = 33;
	  SiPixelRawDataError error(*trailer, errorType);
	  errors.push_back(error);
	}
        trailer++; 
        LogError("PixelDataFormatter::interpretRawData, fedTrailer.check: ")
            <<"fedTrailer.check failed"<<", errorType = 33";
        break; 
      } 
      if ( fedTrailer.lenght()!= nWords) {
        LogError("PROBLEM in PixelDataFormatter,  fedTrailer.lenght()!= nWords !!")<<", errorType = 34";
        if(includeErrors) {
	  int errorType = 34;
	  SiPixelRawDataError error(*trailer, errorType);
	  errors.push_back(error);
	}
      }
      moreTrailers = fedTrailer.moreTrailers();
    }

    // data words
    theWordCounter += 2*(nWords-2);
    LogTrace("")<<"data words: "<< (trailer-header-1);
    for (const Word64* word = header+1; word != trailer; word++) {
      LogTrace("")<<"DATA:    " <<  print(*word);
      static const Word64 WORD32_mask  = 0xffffffff;
      Word32 w1 =  *word >> 32 & WORD32_mask;
      Word32 w2 =  *word       & WORD32_mask;
      if (w2==0) theWordCounter--;

      // check status of word...
      int checkError1 = checkError(w1);
      if (checkError1 != 0) {
	if(includeErrors) {
	  SiPixelRawDataError error1(w1, checkError1);
	  if ((checkError1 == 30)||(checkError1 == 31)) {
	    uint32_t detId1 = errorDetId(converter, w1);
	    error1.setDetId(detId1);
	  }
	  errors.push_back(error1);
	}
      }
      else {
        int status1 = word2digi(converter, w1, digis);
        if (status1) {
	  LogError("PixelDataFormatter::interpretRawData") 
                    << "error #"<<status1<<" returned for word1";
	  switch (status1) {
	    case(1) : {
	      LogError("PixelDataFormatter::interpretRawData")<<"  invalid channel Id (errorType=35)";
	      if(includeErrors) {
		int errorType = 35;
		SiPixelRawDataError error1(w1, errorType);
		errors.push_back(error1);
	      }
	      break;
	    }
            case(2) : {
	      LogError("PixelDataFormatter::interpretRawData")<<"  invalid ROC Id (errorType=36)";
	      if(includeErrors) {
		int errorType = 36;
		SiPixelRawDataError error1(w1, errorType);
		uint32_t detId1 = errorDetId(converter, w1);
		error1.setDetId(detId1);
		errors.push_back(error1);
	      }
	      break;
	    }
            case(3) : {
	      LogError("PixelDataFormatter::interpretRawData")<<"  invalid dcol/pixel value (errorType=37)";
	      if(includeErrors) {
		int errorType = 37;
		SiPixelRawDataError error1(w1, errorType);
		uint32_t detId1 = errorDetId(converter, w1);
		error1.setDetId(detId1);
		errors.push_back(error1);
	      }
	      break;
	    }
	    case(4) : {
	      LogError("PixelDataFormatter::interpretRawData")<<"  dcol/pixel read out of order (errorType=38)";
	      if(includeErrors) {
		int errorType = 38;
		SiPixelRawDataError error1(w1, errorType);
		uint32_t detId1 = errorDetId(converter, w1);
		error1.setDetId(detId1);
		errors.push_back(error1);
	      }
	      break;
	    }
            default: LogError("PixelDataFormatter::interpretRawData")<<"  cabling check returned unexpected result";
	  };
	}
      }
      int checkError2 = checkError(w2);
      if (checkError2 != 0) {
	if(includeErrors) {
	  SiPixelRawDataError error2(w2, checkError2);
	  if ((checkError2 == 30)||(checkError2 == 31)) {
	    uint32_t detId2 = errorDetId(converter, w2);
	    error2.setDetId(detId2);
	  }
	  errors.push_back(error2);
	}
      }
      else {
        int status2 = word2digi(converter, w2, digis);
        if (status2) {
	  LogError("PixelDataFormatter::interpretRawData") 
                    << "error #"<<status2<<" returned for word2";
	  switch (status2) {
	    case(1) : {
	      LogError("PixelDataFormatter::interpretRawData")<<"  invalid channel Id (errorType=35)";
	      if(includeErrors) {
		int errorType = 35;
		SiPixelRawDataError error2(w2, errorType);
		errors.push_back(error2);
	      }
	      break;
	    }
            case(2) : {
	      LogError("PixelDataFormatter::interpretRawData")<<"  invalid ROC Id (errorType=36)";
	      if(includeErrors) {
		int errorType = 36;
		SiPixelRawDataError error2(w2, errorType);
		uint32_t detId2 = errorDetId(converter, w2);
		error2.setDetId(detId2);
		errors.push_back(error2);
	      }
	      break;
	    }
            case(3) : {
	      LogError("PixelDataFormatter::interpretRawData")<<"  invalid dcol/pixel value (errorType=37)";
	      if(includeErrors) {
		int errorType = 37;
		SiPixelRawDataError error2(w2, errorType);
		uint32_t detId2 = errorDetId(converter, w2);
		error2.setDetId(detId2);
		errors.push_back(error2);
	      }
	      break;
	    }
	    case(4) : {
	      LogError("PixelDataFormatter::interpretRawData")<<"  dcol/pixel read out of order (errorType=38)";
	      if(includeErrors) {
		int errorType = 38;
		SiPixelRawDataError error2(w2, errorType);
		uint32_t detId2 = errorDetId(converter, w2);
		error2.setDetId(detId2);
		errors.push_back(error2);
	      }
	      break;
	    }
            default: LogError("PixelDataFormatter::interpretRawData")<<"  cabling check returned unexpected result";
	  };
	}
      }
    }
    
  delete converter;
}


FEDRawData * PixelDataFormatter::formatData(int fedId, const Digis & digis) 
{
  vector<Word32> words;

  static int allDetDigis = 0;
  static int hasDetDigis = 0;
  SiPixelFrameConverter converter(theCablingMap, fedId);
  for (Digis::const_iterator im = digis.begin(); im != digis.end(); im++) {
    allDetDigis++;
//    uint32_t rawId = im->id;
    uint32_t rawId = im->first;
    if ( !converter.hasDetUnit(rawId) ) continue;
    hasDetDigis++;
//    const DetDigis & detDigis = im->data;
    const DetDigis & detDigis = im->second;
    for (DetDigis::const_iterator it = detDigis.begin(); it != detDigis.end(); it++) {
      theDigiCounter++;
      const PixelDigi & digi = (*it);
      int status = digi2word( &converter, rawId, digi, words); 
      if (status) {
         LogError("PixelDataFormatter::formatData exception") 
            <<" digi2word returns error #"<<status
            <<" Ndigis: "<<theDigiCounter << endl
            <<" detector: "<<rawId<< endl
            << print(digi) <<endl; 
      }
    }
  }
  LogTrace(" allDetDigis/hasDetDigis : ") << allDetDigis<<"/"<<hasDetDigis;

  //
  // since digis are written in the form of 64-bit packets
  // add extra 32-bit word to make number of digis even
  //
  if (words.size() %2 != 0) words.push_back( Word32(0) );


  //
  // size in Bytes; create output structure
  //
  int dataSize = words.size() * sizeof(Word32);
  int nHeaders = 1;
  int nTrailers = 1;
  dataSize += (nHeaders+nTrailers)*sizeof(Word64); 
  FEDRawData * rawData = new FEDRawData(dataSize);

  //
  // get begining of data;
  Word64 * word = reinterpret_cast<Word64* >(rawData->data());

  //
  // write one header
  FEDHeader::set(  reinterpret_cast<unsigned char*>(word), 0, 0, 0, fedId); 
  word++;

  //
  // write data
  for (unsigned int i=0; i < words.size(); i+=2) {
    *word = (Word64(words[i]) << 32 ) | words[i+1];
    LogDebug("PixelDataFormatter")  << print(*word);
    word++;
  }

  // write one trailer
  FEDTrailer::set(  reinterpret_cast<unsigned char*>(word), dataSize/sizeof(Word64), 0,0,0);
  word++;

  //
  // check memory
  //
  if (word != reinterpret_cast<Word64* >(rawData->data()+dataSize)) {
    string s = "** PROBLEM in PixelDataFormatter !!!";
    throw cms::Exception(s);
  }

  return rawData;
}

int PixelDataFormatter::checkError(const Word32& data) const
{
 static const Word32 ERROR_mask = ~(~Word32(0) << ROC_bits); 
 int errorType = (data >> ROC_shift) & ERROR_mask;
 switch (errorType) {
    case(25) : {
     LogTrace("")<<"  invalid ROC=25 found (errorType=25)";
     break;
   }
   case(26) : {
     LogTrace("")<<"  gap word found (errorType=26)";
     break;
   }
   case(27) : {
     LogTrace("")<<"  dummy word found (errorType=27)";
     break;
   }
   case(28) : {
     LogTrace("")<<"  error fifo nearly full (errorType=28)";
     break;
   }
   case(29) : {
     LogTrace("")<<"  timeout on a channel (errorType=29)";
     break;
   }
   case(30) : {
     LogTrace("")<<"  trailer error (errorType=30)";
     break;
   }
   case(31) : {
     LogTrace("")<<"  event number error (errorType=31)";
     break;
   }
   default: return 0;
 };
 return errorType;
}

int PixelDataFormatter::digi2word( const SiPixelFrameConverter* converter,
    uint32_t detId, const PixelDigi& digi, std::vector<Word32> & words) const
{
  LogDebug("PixelDataFormatter")
// <<" detId: " << detId 
  <<print(digi);

  SiPixelFrameConverter::DetectorIndex detector = {detId, digi.row(), digi.column()};
  SiPixelFrameConverter::CablingIndex  cabling;
  int status  = converter->toCabling(cabling, detector);
  if (status) return status;

  Word32 word =
             (cabling.link  << LINK_shift)
           | (cabling.roc << ROC_shift)
           | (cabling.dcol << DCOL_shift)
           | (cabling.pxid << PXID_shift)
           | (digi.adc() << ADC_shift);
  words.push_back(word);
  theWordCounter++;
  return 0;
}


int PixelDataFormatter::word2digi(const SiPixelFrameConverter* converter, 
    const Word32 & word, Digis & digis) const
{
  // do not interpret false digis
  if (word == 0 ) return 0;

  static const Word32 LINK_mask = ~(~Word32(0) << LINK_bits);
  static const Word32 ROC_mask  = ~(~Word32(0) << ROC_bits);
  static const Word32 DCOL_mask = ~(~Word32(0) << DCOL_bits);
  static const Word32 PXID_mask = ~(~Word32(0) << PXID_bits);
  static const Word32 ADC_mask  = ~(~Word32(0) << ADC_bits);

  SiPixelFrameConverter::CablingIndex cabling;
  cabling.dcol = (word >> DCOL_shift) & DCOL_mask;
  cabling.pxid = (word >> PXID_shift) & PXID_mask;
  cabling.link = (word >> LINK_shift) & LINK_mask;  
  cabling.roc  = (word >> ROC_shift) & ROC_mask;
  int adc   = (word >> ADC_shift) & ADC_mask;

  static SiPixelFrameConverter::CablingIndex lastcabl;
  static bool lastcablexists = false;

// check to make sure row and dcol values are in order (lowest to highest)
  if (lastcablexists && (lastcabl.roc == cabling.roc) ) {
    if ((cabling.dcol < lastcabl.dcol) || (cabling.dcol==lastcabl.dcol && cabling.pxid < lastcabl.pxid)) {
      LogError("PixelDataFormatter::raw2digi exception") 
              <<" pixel not in correct order (pxid low to high, dcol low to high)"
              <<" link: "<<cabling.link<<", ROC: "<<cabling.roc<<", dcol: "
              <<cabling.dcol<<", pxid: "<<cabling.pxid;
      return 4;
    }
  }
    
  static bool debug = edm::MessageDrop::instance()->debugEnabled;
  if (debug) {
    int rocCol, rocRow;
    sipixelobjects::PixelROC::decodeRowCol(cabling.dcol,cabling.pxid, rocRow, rocCol);
    LogTrace("")<<"  link: "<<cabling.link<<", roc: "<<cabling.roc 
                <<" rocRow: "<<rocRow<<", rocCol:"<<rocCol
                <<" (dcol: "<<cabling.dcol<<",pxid:"<<cabling.pxid<<"), adc:"<<adc;
  }

  if (!converter) return 0;

  SiPixelFrameConverter::DetectorIndex detIdx;
  int status = converter->toDetector(cabling, detIdx);
  if (status) return status; 

  PixelDigi pd(detIdx.row, detIdx.col, adc);
  digis[detIdx.rawId].push_back(pd);

  theDigiCounter++;
  lastcabl = cabling;
  lastcablexists = true;
  if (debug)  LogTrace("") << print(pd);
  return 0;
}

std::string PixelDataFormatter::print(const PixelDigi & digi) const
{
  ostringstream str;
  str << " DIGI: row: " << digi.row() <<", col: " << digi.column() <<", adc: " << digi.adc();
  return str.str();
}

std::string PixelDataFormatter::print(const  Word64 & word) const
{
  ostringstream str;
  //str  <<"word64:  " << *reinterpret_cast<const bitset<64>*> (&word);
  str  <<"word64:  " << reinterpret_cast<const bitset<64>&> (word);
  return str.str();
}

// this function finds the detId for an error word, which cannot be processed in word2digi
uint32_t PixelDataFormatter::errorDetId(const SiPixelFrameConverter* converter, 
    const Word32 & word) const
{
  if (!converter) return 0xffffffff;

  static const Word32 LINK_mask = ~(~Word32(0) << LINK_bits);

  SiPixelFrameConverter::CablingIndex cabling;
  // set dummy values for cabling just to get detId from link
  cabling.dcol = 0;
  cabling.pxid = 0;
  cabling.roc  = 0;
  cabling.link = (word >> LINK_shift) & LINK_mask;   

  SiPixelFrameConverter::DetectorIndex detIdx;
  int status = converter->toDetector(cabling, detIdx);

  return detIdx.rawId;

}
