
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "EventFilter/Utilities/interface/FEDHeader.h"
#include "EventFilter/Utilities/interface/FEDTrailer.h"
#include "EventFilter/Utilities/interface/Crc.h"

#include "EventFilter/ESDigiToRaw/src/ESDataFormatterV4.h"

using namespace std;
using namespace edm;

const int ESDataFormatterV4::bDHEAD    = 2;
const int ESDataFormatterV4::bDH       = 6;
const int ESDataFormatterV4::bDEL      = 24;
const int ESDataFormatterV4::bDERR     = 8;
const int ESDataFormatterV4::bDRUN     = 24;
const int ESDataFormatterV4::bDRUNTYPE = 32;
const int ESDataFormatterV4::bDTRGTYPE = 16;
const int ESDataFormatterV4::bDCOMFLAG = 8;
const int ESDataFormatterV4::bDORBIT   = 32;
const int ESDataFormatterV4::bDVMAJOR  = 8;
const int ESDataFormatterV4::bDVMINOR  = 8;
const int ESDataFormatterV4::bDCH      = 4; 
const int ESDataFormatterV4::bDOPTO    = 8;

const int ESDataFormatterV4::sDHEAD    = 28;
const int ESDataFormatterV4::sDH       = 24;
const int ESDataFormatterV4::sDEL      = 0;
const int ESDataFormatterV4::sDERR     = bDEL + sDEL;
const int ESDataFormatterV4::sDRUN     = 0;
const int ESDataFormatterV4::sDRUNTYPE = 0;
const int ESDataFormatterV4::sDTRGTYPE = 0;
const int ESDataFormatterV4::sDCOMFLAG = bDTRGTYPE + sDTRGTYPE;
const int ESDataFormatterV4::sDORBIT   = 0;
const int ESDataFormatterV4::sDVMINOR  = 8;
const int ESDataFormatterV4::sDVMAJOR  = bDVMINOR + sDVMINOR;
const int ESDataFormatterV4::sDCH      = 0;
const int ESDataFormatterV4::sDOPTO    = 16;

const int ESDataFormatterV4::bKEC    = 8;   // KCHIP packet event counter
const int ESDataFormatterV4::bKFLAG2 = 8;
const int ESDataFormatterV4::bKBC    = 12;  // KCHIP packet bunch counter
const int ESDataFormatterV4::bKFLAG1 = 4;
const int ESDataFormatterV4::bKET    = 1;
const int ESDataFormatterV4::bKCRC   = 1;
const int ESDataFormatterV4::bKCE    = 1;
const int ESDataFormatterV4::bKID    = 16;
const int ESDataFormatterV4::bFIBER  = 6;   // Fiber number
const int ESDataFormatterV4::bKHEAD1 = 2;
const int ESDataFormatterV4::bKHEAD2 = 2;
const int ESDataFormatterV4::bKHEAD  = 4;

const int ESDataFormatterV4::sKEC    = 16;  
const int ESDataFormatterV4::sKFLAG2 = 16; 
const int ESDataFormatterV4::sKBC    = 0;  
const int ESDataFormatterV4::sKFLAG1 = 24; 
const int ESDataFormatterV4::sKET    = 0;
const int ESDataFormatterV4::sKCRC   = bKET + sKET;
const int ESDataFormatterV4::sKCE    = bKCRC + sKCRC;
const int ESDataFormatterV4::sKID    = 0;
const int ESDataFormatterV4::sFIBER  = bKID + sKID + 1;  
const int ESDataFormatterV4::sKHEAD1 = bFIBER + sFIBER + 2;
const int ESDataFormatterV4::sKHEAD2 = bKHEAD1 + sKHEAD1;
const int ESDataFormatterV4::sKHEAD  = 28; 

const int ESDataFormatterV4::bADC0  = 16;
const int ESDataFormatterV4::bADC1  = 16;
const int ESDataFormatterV4::bADC2  = 16;
const int ESDataFormatterV4::bPACE  = 2;
const int ESDataFormatterV4::bSTRIP = 5;
const int ESDataFormatterV4::bE0    = 1;
const int ESDataFormatterV4::bE1    = 1;
const int ESDataFormatterV4::bHEAD  = 4;

const int ESDataFormatterV4::sADC0  = 0;
const int ESDataFormatterV4::sADC1  = bADC0 + sADC0;
const int ESDataFormatterV4::sADC2  = 0;
const int ESDataFormatterV4::sSTRIP = bADC2 + sADC2;
const int ESDataFormatterV4::sPACE  = bSTRIP + sSTRIP; 
const int ESDataFormatterV4::sE0    = bSTRIP + sSTRIP + 1;
const int ESDataFormatterV4::sE1    = bE0 + sE0;
const int ESDataFormatterV4::sHEAD  = 28;

const int ESDataFormatterV4::bOEMUTTCEC = 32; 
const int ESDataFormatterV4::bOEMUTTCBC = 16;
const int ESDataFormatterV4::bOEMUKEC   = 8;
const int ESDataFormatterV4::bOHEAD     = 4;

const int ESDataFormatterV4::sOEMUTTCEC = 0; 
const int ESDataFormatterV4::sOEMUTTCBC = 0;
const int ESDataFormatterV4::sOEMUKEC   = 16;
const int ESDataFormatterV4::sOHEAD     = 28;

ESDataFormatterV4::ESDataFormatterV4(const ParameterSet& ps) 
  : ESDataFormatter(ps) {

  lookup_ = ps.getUntrackedParameter<string>("LookupTable");

  // initialize look-up table
  for (int i=0; i<2; ++i)
    for (int j=0; j<2; ++j)
      for (int k=0 ;k<40; ++k)
        for (int m=0; m<40; m++) {
          fedId_[i][j][k][m] = -1;
	  kchipId_[i][j][k][m] = -1;
          paceId_[i][j][k][m] = -1;
          bundleId_[i][j][k][m] = -1;
          fiberId_[i][j][k][m] = -1;
	}

  // read in look-up table
  int iz, ip, ix, iy, fed, kchip, pace, bundle, fiber;
  ifstream file;
  file.open(lookup_.c_str());
  if( file.is_open() ) {
    for (int i=0; i<4288; ++i) {
      file>> iz >> ip >> ix >> iy >> fed >> kchip >> pace >> bundle >> fiber;
      fedId_[(3-iz)/2-1][ip-1][ix-1][iy-1] = fed;
      kchipId_[(3-iz)/2-1][ip-1][ix-1][iy-1] = kchip;
      paceId_[(3-iz)/2-1][ip-1][ix-1][iy-1] = pace;
      bundleId_[(3-iz)/2-1][ip-1][ix-1][iy-1] = bundle;
      fiberId_[(3-iz)/2-1][ip-1][ix-1][iy-1] = fiber;
    }
  } else {
    cout<<"Look up table file can not be found in "<<lookup_.c_str()<<endl;
  }

}

ESDataFormatterV4::~ESDataFormatterV4() {
}

FEDRawData * ESDataFormatterV4::DigiToRaw(int fedId, const Digis & digis) {

  int ts[3] = {0, 0, 0};
  Word32 word1, word2;
  Word64 word;
  int numberOfStrips = 0 ; 

  map<int, vector<Word64> > map_data;
  vector<Word64> words;
  map_data.clear();

  for (Digis::const_iterator itr = digis.begin(); itr != digis.end(); ++itr) {

    if (itr->first != fedId) continue;
    const DetDigis & detDigis = itr->second;

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
      word2 = (0xc << sHEAD) | (pace << sPACE) | ((detId.strip()-1) << sSTRIP)  | (ts[2] << sADC2);
      word  = (Word64(word2) << 32 ) | Word64(word1);
      
      map_data[kchip].push_back(word);

      // mark global strip number in this FED
      ++numberOfStrips; 

    }
  }

  int iopto; 
  map<int, vector<Word64> >::const_iterator kit;

  kit = map_data.begin(); 

  for(iopto=0; iopto<3; ++iopto) { 
    word2 = (0x6 << sOHEAD) | (kchip_ec_ << sOEMUKEC) | (kchip_bc_ << sOEMUTTCBC) ; 
    word1 = (kchip_ec_ << sOEMUTTCEC) ;
    word  = (Word64(word2) << 32 ) | Word64(word1);
    if (debug_) cout<<"OPTORX: "<<print(word)<<endl; 
    words.push_back(word); 

    int ikchip = 0; 

    while (kit!=map_data.end() && ikchip<12) { // only 12 kchips max per one optorx...
      if (debug_) cout<<"KCHIP : "<<kit->first<<endl;

      word1 = (0 << sKFLAG1) | (0 << sKFLAG2) | (kit->first << sKID);
      word2 = (0x9 << sKHEAD) | (kchip_ec_ << sKEC) | (kchip_bc_ << sKBC); 

      word  = (Word64(word2) << 32 ) | Word64(word1);                                                            

      if (debug_) cout<<"KCHIP : "<<print(word)<<endl; 

      words.push_back(word);           

      const vector<Word64> & data = kit->second; 
      for (unsigned int id=0; id<data.size(); ++id) {
	if (debug_) cout<<"Data  : "<<print(data[id])<<endl;
	words.push_back(data[id]);
      }      
      ++kit ; ++ikchip; 
    } 

  } 

  
//   for (kit=map_data.begin(); kit!=map_data.end(); ++kit) {

//     if (debug_) cout<<"KCHIP : "<<kit->first<<endl;

//     word1 = (0 << sKFLAG1) | (0 << sKFLAG2) | (kit->first << sKID);
//     word2 = (0x9 << sKHEAD) | (kchip_ec_ << sKCE) | (kchip_bc_ << sKBC); 

//     word  = (Word64(word2) << 32 ) | Word64(word1);                                                            

//     if (debug_) cout<<"KCHIP : "<<print(word)<<endl; 

//     words.push_back(word);           

//     const vector<Word64> & data = kit->second; 
//     for (unsigned int id=0; id<data.size(); ++id) {
//       if (debug_) cout<<"Data  : "<<print(data[id])<<endl;
//       words.push_back(data[id]);
//     }

//   } 

  int dataSize = (words.size() + 8) * sizeof(Word64);

  vector<Word64> DCCwords;

  word2 = (3 << sDHEAD) | (1 <<sDH) | (run_number_ << sDRUN);
  word1 = (numberOfStrips << sDEL) | (0xff << sDERR) ;
  word  = (Word64(word2) << 32 ) | Word64(word1);
//   word =  (0ull << (sDHEAD+32)) | (1ull << (sDH+32)) | ((uint64_t)run_number_ << (sDRUN+32)) 
//     | ((uint64_t)numberOfStrips << sDEL);
  DCCwords.push_back(word);

  word2 = (3 << sDHEAD) | (2 <<sDH);
  word1 = 0;
  word  = (Word64(word2) << 32 ) | Word64(word1);
  DCCwords.push_back(word);

  word2 = (3 << sDHEAD) | (3 <<sDH) | (4 << sDVMAJOR) | (0 << sDVMINOR); 
  word1 = (orbit_number_ << sDORBIT);
  word  = (Word64(word2) << 32 ) | Word64(word1);
  DCCwords.push_back(word);

  word2 = (3 << sDHEAD) | (4 <<sDH) | (0x80 << sDOPTO) ; 
  int ich = 0; 
  for(ich=0;ich<4;++ich) word2 |= (0xe  << (ich*4)); // 

  word1 = 0;
  //int chStatus = () ? 0xd : 0xe ; 
  for(ich=0;ich<8;++ich) word1 |= (0xe  << (ich*4)); 

  word  = (Word64(word2) << 32 ) | Word64(word1);
  DCCwords.push_back(word);

  word2 = (3 << sDHEAD) | (5 <<sDH) | (0x80 << sDOPTO);

  for(ich=0;ich<4;++ich) word2 |= (0xe  << (ich*4)); // 

  word1 = 0;
  //int chStatus = () ? 0xd : 0xe ; 
  for(ich=0;ich<8;++ich) word1 |= (0xe  << (ich*4)); 


  word  = (Word64(word2) << 32 ) | Word64(word1);
  DCCwords.push_back(word);

  word2 = (3 << sDHEAD) | (6 <<sDH) | (0x80 << sDOPTO);

  for(ich=0;ich<4;++ich) word2 |= (0xe  << (ich*4)); // 

  word1 = 0;
  //int chStatus = () ? 0xd : 0xe ; 
  for(ich=0;ich<8;++ich) word1 |= (0xe  << (ich*4)); 


  word  = (Word64(word2) << 32 ) | Word64(word1);
  DCCwords.push_back(word);

  // Output (data size in Bytes)
  FEDRawData * rawData = new FEDRawData(dataSize);

  Word64 * w = reinterpret_cast<Word64* >(rawData->data());
  
  // header
  FEDHeader::set( reinterpret_cast<unsigned char*>(w), trgtype_, lv1_, bx_, fedId); 
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
		   evf::compute_crc(rawData->data(), dataSize),
		   0, 0);
  w++;

  return rawData;
}

