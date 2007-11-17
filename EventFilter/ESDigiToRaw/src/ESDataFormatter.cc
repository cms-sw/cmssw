#include "EventFilter/ESDigiToRaw/interface/ESDataFormatter.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "EventFilter/Utilities/interface/FEDHeader.h"
#include "EventFilter/Utilities/interface/FEDTrailer.h"
#include "EventFilter/Utilities/interface/Crc.h"

const int ESDataFormatter::bDHEAD    = 2;
const int ESDataFormatter::bDH       = 6;
const int ESDataFormatter::bDEL      = 24;
const int ESDataFormatter::bDERR     = 8;
const int ESDataFormatter::bDRUN     = 24;
const int ESDataFormatter::bDRUNTYPE = 32;
const int ESDataFormatter::bDTRGTYPE = 16;
const int ESDataFormatter::bDCOMFLAG = 8;
const int ESDataFormatter::bDORBIT   = 32;
const int ESDataFormatter::bDVMAJOR  = 8;
const int ESDataFormatter::bDVMINOR  = 8;
const int ESDataFormatter::bDCH      = 4; 
const int ESDataFormatter::bDOPTO    = 8;

const int ESDataFormatter::sDHEAD    = 26;
const int ESDataFormatter::sDH       = 24;
const int ESDataFormatter::sDEL      = 0;
const int ESDataFormatter::sDERR     = bDEL + sDEL;
const int ESDataFormatter::sDRUN     = 0;
const int ESDataFormatter::sDRUNTYPE = 0;
const int ESDataFormatter::sDTRGTYPE = 0;
const int ESDataFormatter::sDCOMFLAG = bDTRGTYPE + sDTRGTYPE;
const int ESDataFormatter::sDORBIT   = 0;
const int ESDataFormatter::sDVMINOR  = 8;
const int ESDataFormatter::sDVMAJOR  = bDVMINOR + sDVMINOR;
const int ESDataFormatter::sDCH      = 0;
const int ESDataFormatter::sDOPTO    = 16;

const int ESDataFormatter::bKEC    = 8;   // KCHIP packet event counter
const int ESDataFormatter::bKFLAG2 = 8;
const int ESDataFormatter::bKBC    = 12;  // KCHIP packet bunch counter
const int ESDataFormatter::bKFLAG1 = 4;
const int ESDataFormatter::bKET    = 1;
const int ESDataFormatter::bKCRC   = 1;
const int ESDataFormatter::bKCE    = 1;
const int ESDataFormatter::bKID    = 11;
const int ESDataFormatter::bFIBER  = 6;   // Fiber number
const int ESDataFormatter::bKHEAD1 = 2;
const int ESDataFormatter::bKHEAD2 = 2;

const int ESDataFormatter::sKEC    = 0;  
const int ESDataFormatter::sKFLAG2 = bKEC + sKEC;
const int ESDataFormatter::sKBC    = bKFLAG2 + sKFLAG2; 
const int ESDataFormatter::sKFLAG1 = bKBC + sKBC;
const int ESDataFormatter::sKET    = 0;
const int ESDataFormatter::sKCRC   = bKET + sKET;
const int ESDataFormatter::sKCE    = bKCRC + sKCRC;
const int ESDataFormatter::sKID    = bKCE + sKCE + 5;
const int ESDataFormatter::sFIBER  = bKID + sKID + 1;  
const int ESDataFormatter::sKHEAD1 = bFIBER + sFIBER + 2;
const int ESDataFormatter::sKHEAD2 = bKHEAD1 + sKHEAD1;

const int ESDataFormatter::bADC0  = 16;
const int ESDataFormatter::bADC1  = 16;
const int ESDataFormatter::bADC2  = 16;
const int ESDataFormatter::bPACE  = 2;
const int ESDataFormatter::bSTRIP = 5;
const int ESDataFormatter::bE0    = 1;
const int ESDataFormatter::bE1    = 1;
const int ESDataFormatter::bHEAD  = 2;

const int ESDataFormatter::sADC0  = 0;
const int ESDataFormatter::sADC1  = bADC0 + sADC0;
const int ESDataFormatter::sADC2  = 0;
const int ESDataFormatter::sPACE  = bADC2 + sADC2;
const int ESDataFormatter::sSTRIP = bPACE + sPACE; 
const int ESDataFormatter::sE0    = bSTRIP + sSTRIP + 1;
const int ESDataFormatter::sE1    = bE0 + sE0;
const int ESDataFormatter::sHEAD  = bE1 + sE1 + 4;

ESDataFormatter::ESDataFormatter(const ParameterSet& ps) 
  : pset_(ps), run_number_(0), orbit_number_(0), bx_(0), lv1_(0), trgtype_(0)
{

  debug_ = pset_.getUntrackedParameter<bool>("debugMode", false);

}

ESDataFormatter::~ESDataFormatter() {
}

FEDRawData * ESDataFormatter::DigiToRaw(int fedId, const Digis & digis) {

  map<int, vector<Word64> > map_data;
  map_data.clear();  

  int ts[3] = {0, 0, 0};
  Word32 word1, word2;
  Word64 word;
  vector<Word64> words;

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
      word2 = (2 << sHEAD) | (0 << sE1) | (0 << sE0) | ((detId.strip()-1) << sSTRIP) | (pace << sPACE) | (ts[2] << sADC2);
      word  = (Word64(word2) << 32 ) | Word64(word1);
      
      map_data[kchip].push_back(word);
    }
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
  word2 = (0 << sDHEAD) | (1 <<sDH) | (run_number_ << sDRUN);
  word1 = (dataSize << sDEL);
  word  = (Word64(word2) << 32 ) | Word64(word1);
  DCCwords.push_back(word);
  word2 = (0 << sDHEAD) | (2 <<sDH);
  word1 = 0;
  word  = (Word64(word2) << 32 ) | Word64(word1);
  DCCwords.push_back(word);
  word2 = (0 << sDHEAD) | (3 <<sDH) | (1 << sDVMAJOR) | (1 << sDVMINOR); 
  word1 = (orbit_number_ << sDORBIT);
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

FEDRawData * ESDataFormatter::DigiToRawTB(int fedId, const Digis & digis) {
  
  Word8 word8;
  Word16 word16;
  Word32 word1, word2;
  Word64 word;
  vector<Word64> words;

  // Event data
  int data[8][4][32][3];

  for (int i=0; i<8; ++i) 
    for (int j=0; j<4; ++j) 
      for (int k=0; k<32;++k) 
	for (int l=0; l<3; ++l) 
	  data[i][j][k][l] = 1000;
  
  int kchip = 0;
  int pace = 0;
  for (Digis::const_iterator itr = digis.begin(); itr != digis.end(); ++itr) {
    
    if (itr->first != fedId) continue;
    const DetDigis & detDigis = itr->second;
    
    for (DetDigis::const_iterator it = detDigis.begin(); it != detDigis.end(); ++it) {
      
      const ESDataFrame& dataframe = (*it);            
      const ESDetId& detId = dataframe.id();     

      for (int is=0; is<dataframe.size(); ++is) {	

	int ix = detId.six()-30;
	int iy = detId.siy()-19;

	if (ix<2 && iy<2) kchip = 1;
	else if (ix<2 && iy>=2) kchip = 2;
	else if (ix>=2 && iy>=2) kchip = 3;
	else if (ix>=2 && iy<2) kchip = 4;

	if ((ix%2)==0 && (iy%2)==0) pace = 1;
	else if ((ix%2)==0 && (iy%2)==1) pace = 2;
	else if ((ix%2)==1 && (iy%2)==1) pace = 3;
	else if ((ix%2)==1 && (iy%2)==0) pace = 4;

	data[(detId.plane()-1)*4+(kchip-1)][pace-1][detId.strip()-1][is] = dataframe.sample(is).adc();
	cout<<"Digi : "<<detId.plane()<<" "<<detId.six()<<" "<<detId.siy()<<" "<<detId.strip()<<" "<<dataframe.sample(is).adc()<<endl;
      }
    }
  }

  map<int, vector<Word16> > map_data;
  map_data.clear();  

  for (int i=0; i<8; ++i) {

    for (int wl=0; wl<4; ++wl) {      
      if (wl == 0) {
	word16 = (0 << 0); // SOF 
      } else if (wl == 1) {
	word16 = (0 << 0); // FlagI + 12 bit BC
      } else if (wl == 2) {
	word16 = (0 << 0); // FlagII + 8 bit BC
      } else if (wl == 3) {
	word16 = ((i+1) << 0); // KID
      } 
      map_data[i].push_back(word16);    
      if (debug_) cout<<i<<" "<<wl<<" "<<print(word16)<<endl;
    }

    for (int j=0; j<3; ++j) {
      for (int wl=0; wl<34; ++wl) {
	
	if (wl == 0) {
	  word16 = (1 << 8) | 2;  
	  map_data[i].push_back(word16);
	  if (debug_) cout<<i<<" "<<wl<<" "<<print(word16)<<endl;
	} else if (wl == 1) {
	  word16 = (3 << 8) | 4;
	  map_data[i].push_back(word16);
	  if (debug_) cout<<i<<" "<<wl<<" "<<print(word16)<<endl;
	} else {	  
	  word16 = (data[i][0][wl-2][j] << 4) | (data[i][1][wl-2][j] >> 8);
	  map_data[i].push_back(word16);
	  if (debug_) cout<<i<<" "<<wl<<" "<<print(word16)<<endl;
	  word16 = (data[i][1][wl-2][j] << 8) | (data[i][2][wl-2][j] >> 4);
	  map_data[i].push_back(word16);
	  if (debug_) cout<<i<<" "<<wl<<" "<<print(word16)<<endl;
	  word16 = (data[i][2][wl-2][j] << 12) | (data[i][3][wl-2][j] >> 0);
	  map_data[i].push_back(word16);
	  if (debug_) cout<<i<<" "<<wl<<" "<<print(word16)<<endl;
	}
      }
    }

    word16 = 0;
    map_data[i].push_back(word16);
  }

  // Re-organize TB event data 
  vector<Word16> kdata;
  for (int i=0; i<299; ++i) {
   
    // byte1 
    word = 0;
    kdata = map_data[0];
    word = ((Word64(kdata[i]) & 0xff) << 0) | word;
    kdata = map_data[1];
    word = ((Word64(kdata[i]) & 0xff) << 8) | word;	
    kdata = map_data[2];
    word = ((Word64(kdata[i]) & 0xff) << 16) | word;
    kdata = map_data[3];
    word = ((Word64(kdata[i]) & 0xff) << 24) | word;
    kdata = map_data[4];
    word = ((Word64(kdata[i]) & 0xff) << 32) | word;
    kdata = map_data[5];
    word = ((Word64(kdata[i]) & 0xff) << 40) | word;
    words.push_back(word);
    
    word = 0;
    kdata = map_data[6];
    word = ((Word64(kdata[i]) & 0xff) << 0) | word;
    kdata = map_data[7];
    word = ((Word64(kdata[i]) & 0xff) << 8) | word;
    words.push_back(word);    
    
    // byte2
    word = 0;
    kdata = map_data[0];
    word = ((Word64(kdata[i]) & 0xff00) >> 8) | word;
    kdata = map_data[1];
    word = ((Word64(kdata[i]) & 0xff00) >> 0) | word;	
    kdata = map_data[2];
    word = ((Word64(kdata[i]) & 0xff00) << 8) | word;
    kdata = map_data[3];
    word = ((Word64(kdata[i]) & 0xff00) << 16) | word;
    kdata = map_data[4];
    word = ((Word64(kdata[i]) & 0xff00) << 24) | word;
    kdata = map_data[5];
    word = ((Word64(kdata[i]) & 0xff00) << 32) | word;
    words.push_back(word);
    
    word = 0;
    kdata = map_data[6];
    word = ((Word64(kdata[i]) & 0xff00) >> 8) | word;
    kdata = map_data[7];
    word = ((Word64(kdata[i]) & 0xff00) >> 0) | word;
    words.push_back(word);    
  }

  int dataSize = (words.size() + 8) * sizeof(Word64);

  // DCC words
  vector<Word64> DCCwords;
  word2 = (0 << sDHEAD) | (1 <<sDH) | (run_number_ << sDRUN);
  word1 = (dataSize << sDEL);
  word  = (Word64(word2) << 32 ) | Word64(word1);
  DCCwords.push_back(word);
  word2 = (0 << sDHEAD) | (2 <<sDH);
  word1 = 0;
  word  = (Word64(word2) << 32 ) | Word64(word1);
  DCCwords.push_back(word);
  word2 = (0 << sDHEAD) | (3 <<sDH) | (1 << sDVMAJOR) | (1 << sDVMINOR); 
  word1 = (orbit_number_ << sDORBIT);
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
    if (debug_) cout<<"Event data : "<<i<<" "<<print(words[i])<<endl;
    w++;  
  }

  // trailer
  FEDTrailer::set( reinterpret_cast<unsigned char*>(w), dataSize/sizeof(Word64), 
		   evf::compute_crc(rawData->data(), dataSize),
		   0, 0);
  w++;

  return rawData;
}

string ESDataFormatter::print(const  Word64 & word) const
{
  ostringstream str;
  str << "Word64:  " << reinterpret_cast<const bitset<64>&> (word);
  return str.str();
}

string ESDataFormatter::print(const  Word16 & word) const
{
  ostringstream str;
  str << "Word16:  " << reinterpret_cast<const bitset<16>&> (word);
  return str.str();
}
