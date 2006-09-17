#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <bitset>
#include <sstream>

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

void PixelDataFormatter::interpretRawData(int fedId, const FEDRawData& rawData, Digis& digis)
{
  SiPixelFrameConverter converter(theCablingMap, fedId); 
  try {
    int nWords = rawData.size()/8;
    theWordCounter += 2*nWords;
    LogDebug(" PixelDataFormatter ") <<"input size: "<<nWords<<" (8-bytes words)";
    if (nWords !=0) {
      const Word64 * word = reinterpret_cast<const Word64* >(rawData.data());
      for (int i=0; i<nWords; i++) {
        LogDebug("PixelDataFormatter") << print(*word);
        static const Word64 WORD32_mask  = 0xffffffff;
        Word32 w1 =  *word >> 32 & WORD32_mask;
        Word32 w2 =  *word       & WORD32_mask;
        if (w2==0) theWordCounter--;
        word2digi(converter, w1, digis);
        word2digi(converter, w2, digis);
        word++;
      }
    }
  }
  catch ( cms::Exception & err) { LogError("PixelDataFormatter, exception") <<err.what(); }
}


FEDRawData * PixelDataFormatter::formatData(int fedId, const Digis & digis) 
{
  vector<Word32> words;

  static int allDetDigis = 0;
  static int hasDetDigis = 0;
  SiPixelFrameConverter converter(theCablingMap, fedId);
  for (Digis::const_iterator im = digis.begin(); im != digis.end(); im++) {
    allDetDigis++;
    uint32_t rawId = im->first;
    if ( !converter.hasDetUnit(rawId) ) continue;
    hasDetDigis++;
    const DetDigis & detDigis = im->second;
    for (DetDigis::const_iterator it = detDigis.begin(); it != detDigis.end(); it++) {
      theDigiCounter++;
      const PixelDigi & digi = (*it);
      try { digi2word( converter, rawId, digi, words); }
      catch ( cms::Exception& e) { 
         LogError("PixelDataFormatter::formatData exception") <<e.what() 
            <<" Ndigis: "<<theDigiCounter << endl
            <<" detector: "<<rawId<< endl
            << print(digi) <<endl; 
      }
    }
  }
  LogTrace(" allDetDigis/hasDetDigis : ") << allDetDigis<<"/"<<hasDetDigis;

  //
  // since digis are writted in the form og 64-bit packets
  // add extra 32-bit word to make number of digis even
  //
  if (words.size() %2 != 0) words.push_back( Word32(0) );


  //
  // size in Bytes; create output structure
  //
  int dataSize = words.size() * sizeof(Word32);
  if (dataSize == 0) return new FEDRawData(0);
  FEDRawData * rawData = new FEDRawData(dataSize);

  //
  // write data
  //
  Word64 * word = reinterpret_cast<Word64* >(rawData->data());
  for (unsigned int i=0; i < words.size(); i+=2) {
    *word = (Word64(words[i]) << 32 ) | words[i+1];
    LogDebug("PixelDataFormatter")  << print(*word);
    word++;
  }

  //
  // check memory
  //
  if (word != reinterpret_cast<Word64* >(rawData->data()+dataSize)) {
    string s = "** PROBLEM in PixelDataFormatter !!!";
    throw cms::Exception(s);
  }

  return rawData;
}


void PixelDataFormatter::digi2word( const SiPixelFrameConverter& converter,
    uint32_t detId, const PixelDigi& digi, std::vector<Word32> & words) const
{
  LogDebug("PixelDataFormatter")<< print(digi);

  SiPixelFrameConverter::DetectorIndex detector = {detId, digi.row(), digi.column()};
  SiPixelFrameConverter::CablingIndex  cabling = converter.toCabling(detector);

  Word32 word =
             (cabling.link  << LINK_shift)
           | (cabling.roc << ROC_shift)
           | (cabling.dcol << DCOL_shift)
           | (cabling.pxid << PXID_shift)
           | (digi.adc() << ADC_shift);
  words.push_back(word);
  theWordCounter++;
}


void PixelDataFormatter::word2digi(const SiPixelFrameConverter& converter, 
    const Word32 & word, Digis & digis) const
{
  // do not interpret false digis
  if (word == 0 ) return;

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

  SiPixelFrameConverter::DetectorIndex detIdx = converter.toDetector(cabling);

  PixelDigi pd(detIdx.row, detIdx.col,  adc);
  digis[detIdx.rawId].push_back(pd);

  theDigiCounter++;
  LogDebug("PixelDataFormatter") << print(pd);

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
