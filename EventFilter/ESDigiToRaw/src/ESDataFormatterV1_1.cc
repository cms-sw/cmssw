
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "EventFilter/FEDInterface/interface/FEDHeader.h"
#include "EventFilter/FEDInterface/interface/FEDTrailer.h"
#include "FWCore/Utilities/interface/CRC16.h"

#include "EventFilter/ESDigiToRaw/src/ESDataFormatterV1_1.h"

const int ESDataFormatterV1_1::bDHEAD    = 2;
const int ESDataFormatterV1_1::bDH       = 6;
const int ESDataFormatterV1_1::bDEL      = 24;
const int ESDataFormatterV1_1::bDERR     = 8;
const int ESDataFormatterV1_1::bDRUN     = 24;
const int ESDataFormatterV1_1::bDRUNTYPE = 32;
const int ESDataFormatterV1_1::bDTRGTYPE = 16;
const int ESDataFormatterV1_1::bDCOMFLAG = 8;
const int ESDataFormatterV1_1::bDORBIT   = 32;
const int ESDataFormatterV1_1::bDVMAJOR  = 8;
const int ESDataFormatterV1_1::bDVMINOR  = 8;
const int ESDataFormatterV1_1::bDCH      = 4; 
const int ESDataFormatterV1_1::bDOPTO    = 8;

const int ESDataFormatterV1_1::sDHEAD    = 26;
const int ESDataFormatterV1_1::sDH       = 24;
const int ESDataFormatterV1_1::sDEL      = 0;
const int ESDataFormatterV1_1::sDERR     = bDEL + sDEL;
const int ESDataFormatterV1_1::sDRUN     = 0;
const int ESDataFormatterV1_1::sDRUNTYPE = 0;
const int ESDataFormatterV1_1::sDTRGTYPE = 0;
const int ESDataFormatterV1_1::sDCOMFLAG = bDTRGTYPE + sDTRGTYPE;
const int ESDataFormatterV1_1::sDORBIT   = 0;
const int ESDataFormatterV1_1::sDVMINOR  = 8;
const int ESDataFormatterV1_1::sDVMAJOR  = bDVMINOR + sDVMINOR;
const int ESDataFormatterV1_1::sDCH      = 0;
const int ESDataFormatterV1_1::sDOPTO    = 16;

const int ESDataFormatterV1_1::bKEC    = 8;   // KCHIP packet event counter
const int ESDataFormatterV1_1::bKFLAG2 = 8;
const int ESDataFormatterV1_1::bKBC    = 12;  // KCHIP packet bunch counter
const int ESDataFormatterV1_1::bKFLAG1 = 4;
const int ESDataFormatterV1_1::bKET    = 1;
const int ESDataFormatterV1_1::bKCRC   = 1;
const int ESDataFormatterV1_1::bKCE    = 1;
const int ESDataFormatterV1_1::bKID    = 11;
const int ESDataFormatterV1_1::bFIBER  = 6;   // Fiber number
const int ESDataFormatterV1_1::bKHEAD1 = 2;
const int ESDataFormatterV1_1::bKHEAD2 = 2;

const int ESDataFormatterV1_1::sKEC    = 0;  
const int ESDataFormatterV1_1::sKFLAG2 = bKEC + sKEC;
const int ESDataFormatterV1_1::sKBC    = bKFLAG2 + sKFLAG2; 
const int ESDataFormatterV1_1::sKFLAG1 = bKBC + sKBC;
const int ESDataFormatterV1_1::sKET    = 0;
const int ESDataFormatterV1_1::sKCRC   = bKET + sKET;
const int ESDataFormatterV1_1::sKCE    = bKCRC + sKCRC;
const int ESDataFormatterV1_1::sKID    = bKCE + sKCE + 5;
const int ESDataFormatterV1_1::sFIBER  = bKID + sKID + 1;  
const int ESDataFormatterV1_1::sKHEAD1 = bFIBER + sFIBER + 2;
const int ESDataFormatterV1_1::sKHEAD2 = bKHEAD1 + sKHEAD1;

const int ESDataFormatterV1_1::bADC0  = 16;
const int ESDataFormatterV1_1::bADC1  = 16;
const int ESDataFormatterV1_1::bADC2  = 16;
const int ESDataFormatterV1_1::bPACE  = 2;
const int ESDataFormatterV1_1::bSTRIP = 5;
const int ESDataFormatterV1_1::bE0    = 1;
const int ESDataFormatterV1_1::bE1    = 1;
const int ESDataFormatterV1_1::bHEAD  = 2;

const int ESDataFormatterV1_1::sADC0  = 0;
const int ESDataFormatterV1_1::sADC1  = bADC0 + sADC0;
const int ESDataFormatterV1_1::sADC2  = 0;
const int ESDataFormatterV1_1::sPACE  = bADC2 + sADC2;
const int ESDataFormatterV1_1::sSTRIP = bPACE + sPACE; 
const int ESDataFormatterV1_1::sE0    = bSTRIP + sSTRIP + 1;
const int ESDataFormatterV1_1::sE1    = bE0 + sE0;
const int ESDataFormatterV1_1::sHEAD  = bE1 + sE1 + 4;

using namespace std; 
using namespace edm; 


ESDataFormatterV1_1::ESDataFormatterV1_1(const ParameterSet& ps) 
  : ESDataFormatter(ps) {
}

ESDataFormatterV1_1::~ESDataFormatterV1_1() {
}

// FEDRawData * ESDataFormatterV1_1::DigiToRawDCC(int fedId, const Digis & digis) {

//   int ts[3] = {0, 0, 0};
//   Word32 word1, word2;
//   Word64 word;

//   for (Digis::const_iterator itr = digis.begin(); itr != digis.end(); ++itr) {

//     if (itr->first != fedId) continue;
//     const DetDigis & detDigis = itr->second;

//     for (DetDigis::const_iterator it = detDigis.begin(); it != detDigis.end(); ++it) {

//       const ESDataFrame& dataframe = (*it);
// //       const ESDetId& detId = dataframe.id();

//       for (int is=0; is<dataframe.size(); ++is) ts[is] = dataframe.sample(is).adc();
//     }
//   }

//   // DCC words
//   vector<Word64> DCCwords;
//   word2 = (run_number_ << 0) ;
//   word1 = 0;
//   word =  (Word64(word2) << 32 ) | Word64(word1);

//   DCCwords.push_back(word);
//   for (int i=0; i<5; ++i) {
//     word = 0;
//     DCCwords.push_back(word);
//   }

//   FEDRawData * rawData = new FEDRawData(0);

//   return rawData;
// }

void ESDataFormatterV1_1::DigiToRaw(int fedId, Digis& digis, FEDRawData& fedRawData, const Meta_Data & meta_data) const{

  map<int, vector<Word64> > map_data;
  map_data.clear();  

  int ts[3] = {0, 0, 0};
  Word32 word1, word2;
  Word64 word;
  vector<Word64> words;


    const DetDigis& detDigis = digis[fedId];

//     if (detDigis==digis.end()) { 
//       cout << "ESDataFormatterV1_1::DigiToRaw : could not find digi vector in digis map for fedID: " 
// 	   << fedId << endl ; 
//       return 0; 
//     } 

    for (DetDigis::const_iterator it = detDigis.begin(); it != detDigis.end(); ++it) {
      
      const ESDataFrame& dataframe = (*it);            
      const ESDetId& detId = dataframe.id();     
      
      for (int is=0; is<dataframe.size(); ++is) ts[is] = dataframe.sample(is).adc();            
      
      //  calculate fake kchip and pace id 
      int kchip = -1;
      int pace = -1;
      int ix = -1;
      int iy = -1;
      
      ix = detId.six() % 2;
      iy = detId.siy() % 2;
      if (ix == 1 && iy == 1)
	pace = 0;
      else if (ix == 0 && iy == 1)
	pace = 1;
      else if (ix == 1 && iy == 0) 
	pace = 2;
      else if (ix == 0 && iy == 0)
	pace = 3;
      
      ix = (1 + detId.six()) / 2;
      iy = (1 + detId.siy()) / 2;  
      if (detId.zside() == 1 && detId.plane() == 1) 
	kchip = ix + (iy-1)*20 - 1;
      else if (detId.zside() == 1 && detId.plane() == 2) 
	kchip = ix + (iy-1)*20 + 399;
      else if (detId.zside() == -1 && detId.plane() == 1) 
	kchip = ix + (iy-1)*20 + 799;
      else if (detId.zside() == -1 && detId.plane() == 2) 
	kchip = ix + (iy-1)*20 + 1199;
      
      if (debug_) cout<<"Si : "<<detId.zside()<<" "<<detId.plane()<<" "<<detId.six()<<" "<<detId.siy()<<" "<<detId.strip()<<" ("<<kchip<<","<<pace<<") "<<ts[2]<<" "<<ts[1]<<" "<<ts[0]<<endl;

      word1 = (ts[1] << sADC1) | (ts[0] << sADC0);
      word2 = (2 << sHEAD) | (0 << sE1) | (0 << sE0) | ((detId.strip()-1) << sSTRIP) | (pace << sPACE) | (ts[2] << sADC2);
      word  = (Word64(word2) << 32 ) | Word64(word1);
      
      map_data[kchip].push_back(word);
    }


  map<int, vector<Word64> >::const_iterator kit;
  for (kit=map_data.begin(); kit!=map_data.end(); ++kit) {

    if (debug_) cout<<"KCHIP : "<<kit->first<<endl;

    word1 = (0 << sKFLAG1) | (0 << sKBC) | (0 << sKFLAG2) | (0 << sKEC);                                                       
    word2 = (1 << sKHEAD2) | (0 << sKHEAD1) | (0 << sFIBER) | (kit->first << sKID) | (0 << sKCE) | (0 << sKCRC) | (0 << sKET);
    word  = (Word64(word2) << 32 ) | Word64(word1);                                                                            

    if (debug_) cout<<"KCHIP : "<<print(word)<<endl; 
    words.push_back(word);           

    const vector<Word64> & data = kit->second; 
    for (unsigned int id=0; id<data.size(); ++id) {
      if (debug_) cout<<"Data  : "<<print(data[id])<<endl;
      words.push_back(data[id]);
    }

  } 

  int dataSize = (words.size() + 8) * sizeof(Word64);
  
  // DCC words
  vector<Word64> DCCwords;
  word2 = (0 << sDHEAD) | (1 <<sDH) | (meta_data.run_number << sDRUN);
  word1 = (dataSize << sDEL);
  word  = (Word64(word2) << 32 ) | Word64(word1);
  DCCwords.push_back(word);
  word2 = (0 << sDHEAD) | (2 <<sDH);
  word1 = 0;
  word  = (Word64(word2) << 32 ) | Word64(word1);
  DCCwords.push_back(word);
  word2 = (0 << sDHEAD) | (3 <<sDH) | (1 << sDVMAJOR) | (1 << sDVMINOR); 
  word1 = (meta_data.orbit_number << sDORBIT);
  word  = (Word64(word2) << 32 ) | Word64(word1);
  DCCwords.push_back(word);
  word2 = (0 << sDHEAD) | (4 <<sDH);
  word1 = 0;
  word  = (Word64(word2) << 32 ) | Word64(word1);
  DCCwords.push_back(word);
  word2 = (0 << sDHEAD) | (5 <<sDH);
  word1 = 0;
  word  = (Word64(word2) << 32 ) | Word64(word1);
  DCCwords.push_back(word);
  word2 = (0 << sDHEAD) | (6 <<sDH);
  word1 = 0;
  word  = (Word64(word2) << 32 ) | Word64(word1);
  DCCwords.push_back(word);
  
  // Output (data size in Bytes)
  fedRawData.resize(dataSize);

  Word64 * w = reinterpret_cast<Word64* >(fedRawData.data());
  
  // header
  FEDHeader::set( reinterpret_cast<unsigned char*>(w), trgtype_, meta_data.lv1, meta_data.bx, fedId); 
  w++;

  // ES-DCC 
  for (unsigned int i=0; i<DCCwords.size(); ++i) {
    if (debug_) cout<<"DCC  : "<<print(DCCwords[i])<<endl;
    *w = DCCwords[i];
    w++;
  }

  // event data
  for (unsigned int i=0; i<words.size(); ++i) {
    *w = words[i];
    w++;  
  }

  // trailer
  FEDTrailer::set( reinterpret_cast<unsigned char*>(w), dataSize/sizeof(Word64), 
		   evf::compute_crc(fedRawData.data(), dataSize),
		   0, 0);
  w++;
 

}

