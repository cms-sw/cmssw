//-------------------------------------------------
//
//   Class: DTTFFEDSim
//
//   L1 DT Track Finder Digi-to-Raw
//
//
//   $Date: 2007/03/12 00:44:19 $
//   $Revision: 1.3 $
//
//   Author :
//   J. Troconiz  UAM Madrid
//
//--------------------------------------------------

#include "EventFilter/DTTFRawToDigi/interface/DTTFFEDSim.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h>

#include <iostream>
#include <string>

using namespace std;

DTTFFEDSim::DTTFFEDSim(const edm::ParameterSet& pset) : eventNum(0) {

  produces<FEDRawDataCollection>();

}

DTTFFEDSim::~DTTFFEDSim(){}

void DTTFFEDSim::produce(edm::Event& e, const edm::EventSetup& c) {

  FEDRawDataCollection data;

  if (!fillRawData(e, data)) return;

  auto_ptr<FEDRawDataCollection> fed_product(new FEDRawDataCollection(data));

  e.put(fed_product);

}

bool DTTFFEDSim::fillRawData(edm::Event& e,
                             FEDRawDataCollection& data) {

  eventNum++;

  int lines = 2;

  edm::Handle<L1MuDTChambPhContainer> dttrig;
  e.getByType(dttrig);
  lines += dttrig->bxSize(-1, 1);

  FEDRawData& dttfdata = data.FEDData(0x30C);
  dttfdata.resize(lines*8); // size in bytes
  unsigned char* LineFED=dttfdata.data();

  long* dataWord1 = new long;
  long* dataWord2 = new long;

  //--> Header

  *dataWord1 = 0x50000000
             + (eventNum&0xFFFFFF);
  *dataWord2 = 0x00030C00;

  int newCRC =  0xFFFF;
  calcCRC(*dataWord1, *dataWord2, newCRC);
  *((long*)LineFED)=*dataWord2; 
  LineFED+=4;
  *((long*)LineFED)=*dataWord1; 

  //--> DTTF data 

  int TS1Id[4], TS2Id[4];   // word identifier for TS #1,#2 for stations 
  TS1Id[0] = 0x0E;      
  TS2Id[0] = 0x1E;
  TS1Id[1] = 0x2E;
  TS2Id[1] = 0x3E;
  TS1Id[3] = 0x4E;
  TS2Id[3] = 0x5E;
  TS1Id[2] = 0x8FFF8;
  TS2Id[2] = 0x9FFF8;

  //Input
  L1MuDTChambPhContainer::Phi_iterator tsphi;

  for ( tsphi =  dttrig->getContainer()->begin();
        tsphi != dttrig->getContainer()->end();
        tsphi++ ) {
    if ( tsphi->code() != 7 ) {

      int wheelID   = tsphi->whNum()+1;
      if ( wheelID <= 0 ) wheelID -= 2;
      int stationID = tsphi->stNum()-1;
      int is2nd     = ( tsphi->Ts2Tag() == 1 ) ? 1 : 0;

      int channelNr = channel(wheelID, tsphi->scNum(), tsphi->bxNum()-is2nd);
      if ( channelNr == 255 ) continue;
      int TSId = ( is2nd == 0 ) ? TS1Id[stationID] : TS2Id[stationID];

      *dataWord1 = ((channelNr&0xFF)<<24)
                 + 0x00FFFFFF;

      if ( stationID != 2 ){
        *dataWord2 = (             (TSId&0x0FF)<<24)
	           + (~(tsphi->code()+1)&0x007)
	           + (   (~tsphi->phiB()&0x3FF)<<3) 
	           + (    (~tsphi->phi()&0xFFF)<<13);
      }
      else {
        *dataWord2 = (             (TSId&0xFFFFF)<<12) 
	           + (~(tsphi->code()+1)&0x00007)
	           + (    (~tsphi->phi()&0x00FFF)<<3);
      }

      calcCRC(*dataWord1, *dataWord2, newCRC);
      LineFED+=4;
      *((long*)LineFED)=*dataWord2; 
      LineFED+=4;
      *((long*)LineFED)=*dataWord1; 
    }
  }
  //Input

  //--> Trailer

  *dataWord1 = 0xA0000000
             + (lines&0xFFFFFF);
  *dataWord2 = 0;

  calcCRC(*dataWord1, *dataWord2&0xFFFF, newCRC);

  *dataWord2 += (newCRC&0xFFFF)<<16;

  LineFED+=4;
  *((long*)LineFED)=*dataWord2; 
  LineFED+=4;
  *((long*)LineFED)=*dataWord1; 

  delete dataWord1;
  delete dataWord2;
  return true;
}

int DTTFFEDSim::channel( int wheel, int sector,  int bx ){

  // wheel  :  -3 -2 -1 +1 +2 +3 <=> PHTF's : N2, N1, N0, P0, P1, P2
  //                           0 <=> ETTF  
  // sector :  0 -> 11
  // bx     : -1 -> +1

  int myChannel = 255;

  if ( abs(bx) > 1)               { return myChannel; }
  if ( sector < 0 || sector > 11) { return myChannel; }
  if ( abs(wheel) > 3)            { return myChannel; }

   myChannel = sector*21 + wheel*3 + bx + 10 ; 

  return myChannel;
}

int DTTFFEDSim::bxNr( int channel ){

  if (channel < 0 || channel > 252 ){ return -999; }

  int myBx = (channel%3)-1;

  return myBx;
}

int DTTFFEDSim::sector( int channel ){

  if (channel < 0 || channel > 252 ){ return -999; }

  return channel/21;
}

int DTTFFEDSim::wheel( int channel ){

  if (channel < 0 || channel > 252 ){ return -999; }    

  int myWheel = ((channel%21)/3)-3;

  return myWheel;
}

void DTTFFEDSim::calcCRC(long myD1, long myD2, int &myC){

  int myCRC[16],D[64],C[16];

  for( int i=0; i < 32; i++ ){ D[i]=(myD2>>i)&0x1; }
  for( int i=0; i < 32; i++ ){ D[i+32]=(myD1>>i)&0x1; }
  for( int i=0; i < 16; i++ ){ C[i]=(myC>>i)&0x1; }

  myCRC[0] = ( D[63] + D[62] + D[61] + D[60] + D[55] + D[54] +
               D[53] + D[52] + D[51] + D[50] + D[49] + D[48] +
               D[47] + D[46] + D[45] + D[43] + D[41] + D[40] +
               D[39] + D[38] + D[37] + D[36] + D[35] + D[34] +
               D[33] + D[32] + D[31] + D[30] + D[27] + D[26] +
               D[25] + D[24] + D[23] + D[22] + D[21] + D[20] +
               D[19] + D[18] + D[17] + D[16] + D[15] + D[13] +
               D[12] + D[11] + D[10] + D[9]  + D[8]  + D[7]  +
               D[6]  + D[5]  + D[4]  + D[3]  + D[2]  + D[1]  +
               D[0]  + C[0]  + C[1]  + C[2]  + C[3]  + C[4]  +
               C[5]  + C[6]  + C[7]  + C[12] + C[13] + C[14] +
               C[15] )%2;

  myCRC[1] = ( D[63] + D[62] + D[61] + D[56] + D[55] + D[54] +
               D[53] + D[52] + D[51] + D[50] + D[49] + D[48] +
               D[47] + D[46] + D[44] + D[42] + D[41] + D[40] +
               D[39] + D[38] + D[37] + D[36] + D[35] + D[34] +
               D[33] + D[32] + D[31] + D[28] + D[27] + D[26] +
	       D[25] + D[24] + D[23] + D[22] + D[21] + D[20] +
	       D[19] + D[18] + D[17] + D[16] + D[14] + D[13] +
	       D[12] + D[11] + D[10] + D[9]  + D[8]  + D[7]  +
	       D[6]  + D[5]  + D[4]  + D[3]  + D[2]  + D[1]  +
	       C[0]  + C[1]  + C[2]  + C[3]  + C[4]  + C[5]  +
	       C[6]  + C[7]  + C[8]  + C[13] + C[14] + C[15] )%2;

  myCRC[2] = ( D[61] + D[60] + D[57] + D[56] + D[46] + D[42] +
	       D[31] + D[30] + D[29] + D[28] + D[16] + D[14] +
	       D[1]  + D[0]  + C[8]  + C[9]  + C[12] + C[13] )%2;

  myCRC[3] = ( D[62] + D[61] + D[58] + D[57] + D[47] + D[43] +
	       D[32] + D[31] + D[30] + D[29] + D[17] + D[15] +
	       D[2]  + D[1]  + C[9]  + C[10] + C[13] + C[14] )%2;

  myCRC[4] = ( D[63] + D[62] + D[59] + D[58] + D[48] + D[44] +
	       D[33] + D[32] + D[31] + D[30] + D[18] + D[16] + 
	       D[3]  + D[2]  + C[0]  + C[10] + C[11] + C[14] +
	       C[15] )%2;

  myCRC[5] = ( D[63] + D[60] + D[59] + D[49] + D[45] + D[34] +
	       D[33] + D[32] + D[31] + D[19] + D[17] + D[4]  +
	       D[3]  + C[1]  + C[11] + C[12] + C[15] )%2;

  myCRC[6] = ( D[61] + D[60] + D[50] + D[46] + D[35] + D[34] +
	       D[33] + D[32] + D[20] + D[18] + D[5]  + D[4]   +
	       C[2]  + C[12] + C[13] )%2;

  myCRC[7] = ( D[62] + D[61] + D[51] + D[47] + D[36] + D[35] +
	       D[34] + D[33] + D[21] + D[19] + D[6]  + D[5]   +
	       C[3]  + C[13] + C[14] )%2;

  myCRC[8] = ( D[63] + D[62] + D[52] + D[48] + D[37] + D[36] +
	       D[35] + D[34] + D[22] + D[20] + D[7]  + D[6]   +
	       C[0]  + C[4]  + C[14] + C[15] )%2;

  myCRC[9] = ( D[63] + D[53] + D[49] + D[38] + D[37] + D[36] +
	       D[35] + D[23] + D[21] + D[8]  + D[7]  + C[1]  +
	       C[5]  + C[15] )%2;

  myCRC[10] = ( D[54] + D[50] + D[39] + D[38] + D[37] + D[36] + 
		D[24] + D[22] + D[9]  + D[8]  + C[2]  + C[6] )%2;

  myCRC[11] = ( D[55] + D[51] + D[40] + D[39] + D[38] + D[37] +
		D[25] + D[23] + D[10] + D[9]  + C[3]  + C[7] )%2;

  myCRC[12] = ( D[56] + D[52] + D[41] + D[40] + D[39] + D[38] +
		D[26] + D[24] + D[11] + D[10] + C[4] + C[8] )%2;

  myCRC[13] = ( D[57] + D[53] + D[42] + D[41] + D[40] + D[39] +
		D[27] + D[25] + D[12] + D[11] + C[5]  + C[9] )%2;

  myCRC[14] = ( D[58] + D[54] + D[43] + D[42] + D[41] + D[40] +
		D[28] + D[26] + D[13] + D[12] + C[6]  + C[10] )%2;

  myCRC[15] = ( D[63] + D[62] + D[61] + D[60] + D[59] + D[54] +
		D[53] + D[52] + D[51] + D[50] + D[49] + D[48] + 
		D[47] + D[46] + D[45] + D[44] + D[42] + D[40] +
		D[39] + D[38] + D[37] + D[36] + D[35] + D[34] + 
		D[33] + D[32] + D[31] + D[30] + D[29] + D[26] +
		D[25] + D[24] + D[23] + D[22] + D[21] + D[20] + 
		D[19] + D[18] + D[17] + D[16] + D[15] + D[14] +
		D[12] + D[11] + D[10] + D[9]  + D[8]  + D[7]  + 
		D[6]  + D[5]  + D[4]  + D[3]  + D[2]  + D[1]  +
		D[0]  + C[0]  + C[1]  + C[2]  + C[3]  + C[4]  + 
		C[5]  + C[6]  + C[11] + C[12] + C[13] + C[14] +
		C[15] )%2;

  int tempC = 0x0;  
  for(int i=0; i<16 ; i++){ tempC = tempC + (myCRC[i]<<i); }
  myC = tempC ;
  return;
}
