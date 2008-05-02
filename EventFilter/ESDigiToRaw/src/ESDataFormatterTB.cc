

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "EventFilter/Utilities/interface/FEDHeader.h"
#include "EventFilter/Utilities/interface/FEDTrailer.h"
#include "EventFilter/Utilities/interface/Crc.h"

#include "EventFilter/ESDigiToRaw/src/ESDataFormatterTB.h"

const int ESDataFormatterTB::bDHEAD    = 2;
const int ESDataFormatterTB::bDH       = 6;
const int ESDataFormatterTB::bDEL      = 24;
const int ESDataFormatterTB::bDERR     = 8;
const int ESDataFormatterTB::bDRUN     = 24;
const int ESDataFormatterTB::bDRUNTYPE = 32;
const int ESDataFormatterTB::bDTRGTYPE = 16;
const int ESDataFormatterTB::bDCOMFLAG = 8;
const int ESDataFormatterTB::bDORBIT   = 32;
const int ESDataFormatterTB::bDVMAJOR  = 8;
const int ESDataFormatterTB::bDVMINOR  = 8;
const int ESDataFormatterTB::bDCH      = 4; 
const int ESDataFormatterTB::bDOPTO    = 8;

const int ESDataFormatterTB::sDHEAD    = 26;
const int ESDataFormatterTB::sDH       = 24;
const int ESDataFormatterTB::sDEL      = 0;
const int ESDataFormatterTB::sDERR     = bDEL + sDEL;
const int ESDataFormatterTB::sDRUN     = 0;
const int ESDataFormatterTB::sDRUNTYPE = 0;
const int ESDataFormatterTB::sDTRGTYPE = 0;
const int ESDataFormatterTB::sDCOMFLAG = bDTRGTYPE + sDTRGTYPE;
const int ESDataFormatterTB::sDORBIT   = 0;
const int ESDataFormatterTB::sDVMINOR  = 8;
const int ESDataFormatterTB::sDVMAJOR  = bDVMINOR + sDVMINOR;
const int ESDataFormatterTB::sDCH      = 0;
const int ESDataFormatterTB::sDOPTO    = 16;

const int ESDataFormatterTB::bKEC    = 8;   // KCHIP packet event counter
const int ESDataFormatterTB::bKFLAG2 = 8;
const int ESDataFormatterTB::bKBC    = 12;  // KCHIP packet bunch counter
const int ESDataFormatterTB::bKFLAG1 = 4;
const int ESDataFormatterTB::bKET    = 1;
const int ESDataFormatterTB::bKCRC   = 1;
const int ESDataFormatterTB::bKCE    = 1;
const int ESDataFormatterTB::bKID    = 11;
const int ESDataFormatterTB::bFIBER  = 6;   // Fiber number
const int ESDataFormatterTB::bKHEAD1 = 2;
const int ESDataFormatterTB::bKHEAD2 = 2;

const int ESDataFormatterTB::sKEC    = 0;  
const int ESDataFormatterTB::sKFLAG2 = bKEC + sKEC;
const int ESDataFormatterTB::sKBC    = bKFLAG2 + sKFLAG2; 
const int ESDataFormatterTB::sKFLAG1 = bKBC + sKBC;
const int ESDataFormatterTB::sKET    = 0;
const int ESDataFormatterTB::sKCRC   = bKET + sKET;
const int ESDataFormatterTB::sKCE    = bKCRC + sKCRC;
const int ESDataFormatterTB::sKID    = bKCE + sKCE + 5;
const int ESDataFormatterTB::sFIBER  = bKID + sKID + 1;  
const int ESDataFormatterTB::sKHEAD1 = bFIBER + sFIBER + 2;
const int ESDataFormatterTB::sKHEAD2 = bKHEAD1 + sKHEAD1;

const int ESDataFormatterTB::bADC0  = 16;
const int ESDataFormatterTB::bADC1  = 16;
const int ESDataFormatterTB::bADC2  = 16;
const int ESDataFormatterTB::bPACE  = 2;
const int ESDataFormatterTB::bSTRIP = 5;
const int ESDataFormatterTB::bE0    = 1;
const int ESDataFormatterTB::bE1    = 1;
const int ESDataFormatterTB::bHEAD  = 2;

const int ESDataFormatterTB::sADC0  = 0;
const int ESDataFormatterTB::sADC1  = bADC0 + sADC0;
const int ESDataFormatterTB::sADC2  = 0;
const int ESDataFormatterTB::sPACE  = bADC2 + sADC2;
const int ESDataFormatterTB::sSTRIP = bPACE + sPACE; 
const int ESDataFormatterTB::sE0    = bSTRIP + sSTRIP + 1;
const int ESDataFormatterTB::sE1    = bE0 + sE0;
const int ESDataFormatterTB::sHEAD  = bE1 + sE1 + 4;

using namespace std; 
using namespace edm; 


ESDataFormatterTB::ESDataFormatterTB(const ParameterSet& ps) 
  : ESDataFormatter(ps) {
}

ESDataFormatterTB::~ESDataFormatterTB() {
}



FEDRawData * ESDataFormatterTB::DigiToRaw(int fedId, Digis & digis) {
  
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


