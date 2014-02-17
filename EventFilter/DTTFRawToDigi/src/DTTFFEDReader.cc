//-------------------------------------------------
//
//   Class: DTTFFEDReader
//
//   L1 DT Track Finder Raw-to-Digi
//
//
//   $Date: 2009/11/18 13:27:12 $
//   $Revision: 1.16 $
//
//   Author :
//   J. Troconiz  UAM Madrid
//
//--------------------------------------------------

#include "EventFilter/DTTFRawToDigi/interface/DTTFFEDReader.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include <iostream>

using namespace std;


DTTFFEDReader::DTTFFEDReader(const edm::ParameterSet& pset) {

  produces<L1MuDTChambPhContainer>();
  produces<L1MuDTChambThContainer>();
  produces<L1MuDTTrackContainer>("DATA");

  DTTFInputTag = pset.getParameter<edm::InputTag>("DTTF_FED_Source");

  verbose_ =  pset.getUntrackedParameter<bool>("verbose",false);

}

DTTFFEDReader::~DTTFFEDReader(){}

void DTTFFEDReader::produce(edm::Event& e, const edm::EventSetup& c) {

  auto_ptr<L1MuDTChambPhContainer> phi_product(new L1MuDTChambPhContainer);
  auto_ptr<L1MuDTChambThContainer> the_product(new L1MuDTChambThContainer);
  auto_ptr<L1MuDTTrackContainer>   tra_product(new L1MuDTTrackContainer);

  L1MuDTChambPhContainer::Phi_Container phi_data;
  L1MuDTChambThContainer::The_Container the_data;
  L1MuDTTrackContainer::TrackContainer  tra_data;

  if (!fillRawData(e, phi_data, the_data, tra_data)) return;

  phi_product->setContainer(phi_data);
  the_product->setContainer(the_data);
  tra_product->setContainer(tra_data);

  e.put(phi_product);
  e.put(the_product);
  e.put(tra_product,"DATA");

}

bool DTTFFEDReader::fillRawData(edm::Event& e,
                                L1MuDTChambPhContainer::Phi_Container& phi_data,
                                L1MuDTChambThContainer::The_Container& the_data,
                                L1MuDTTrackContainer::TrackContainer&  tra_data) {

  analyse(e);

  phi_data = p_data();
  the_data = t_data();
  tra_data = k_data();

  return true;
}

//--------------
// Operations --
//--------------
void DTTFFEDReader::analyse(edm::Event& e) {
  clear();
  process(e);
  match();
  return;
}

// process data
void DTTFFEDReader::process(edm::Event& e) {

  // Container
  vector<int> DTTFWordContainer; 
  vector<int>::iterator DTTFiterator;

  // Header constituents
  int BOEevTy, DTTFId;

  // DTTF Payload constituents 
  int DTTFWord;
  int DTTFChan, bitsID;
  int addr1[2] = { 3,  3};
  int addr2[2] = {15, 15};
  int addr3[2] = {15, 15};
  int addr4[2] = {15, 15};

  // Trailer constituents
  int evtLgth , CRC;

  //--> Header

  edm::Handle<FEDRawDataCollection> data;
  e.getByLabel(getDTTFInputTag(),data);
  FEDRawData dttfdata = data->FEDData(0x030C);
  if ( dttfdata.size() == 0 ) return;

  int* dataWord1 = new int;
  int* dataWord2 = new int;
  unsigned char* LineFED=dttfdata.data();
  *dataWord2=*((int*)LineFED);
  LineFED+=4;
  *dataWord1=*((int*)LineFED);
  int lines  = 1; // already counting header

  BOEevTy = ((*dataWord1)&0xFF000000)>>24; // positions 57 ->64
  DTTFId  = ((*dataWord2)&0x000FFF00)>>8;  // positions 9 ->20

  if( (BOEevTy != 0x50) || ( DTTFId != 0x030C) ){
    if ( verbose_ ) edm::LogWarning("dttf_unpacker")
      << "Not a DTTF header " << hex << *dataWord1;
    delete dataWord1;
    delete dataWord2;
    return;
  }

  int newCRC =  0xFFFF;
  calcCRC(*dataWord1, *dataWord2, newCRC);  


  //--> DTTF data 

  LineFED+=4;
  *dataWord2=*((int*)LineFED);
  LineFED+=4;
  *dataWord1=*((int*)LineFED);
  int chkEOE = ((*dataWord1)&0xFFF00000)>>20; 
  lines++;

  while(chkEOE != 0xA00){

    calcCRC(*dataWord1, *dataWord2, newCRC);

    DTTFWord = *dataWord1;
    DTTFWordContainer.push_back(DTTFWord);
    DTTFWord = *dataWord2;
    DTTFWordContainer.push_back(DTTFWord);

    LineFED+=4;
    *dataWord2=*((int*)LineFED);
    LineFED+=4;
    *dataWord1=*((int*)LineFED);
    chkEOE = ((*dataWord1)&0xFFF00000)>>20; 
    lines++;

    if(lines > 3026){
      if ( verbose_ ) edm::LogWarning("dttf_unpacker")
        << "Warning : number of DTTF lines > 3026 "; // 3026 = 1(header) + 3024(max # PHTF-ETTF 64 bits words) + 1(trailer)
      delete dataWord1;
      delete dataWord2;
      return;
    }

  } // end while-Data loop


  //--> Trailer

  evtLgth   = ((*dataWord1)&0x00FFFFFF);     // positions 33 ->56
  CRC       = ((*dataWord2)&0xFFFF0000)>>16; // positions 17 ->32

  calcCRC(*dataWord1, (*dataWord2)&0xFFFF, newCRC);
  if( newCRC != CRC){
    if ( verbose_ ) edm::LogWarning("dttf_unpacker")
      << "Calculated CRC " << hex << newCRC << " differs from CRC in trailer " << hex << CRC;
    delete dataWord1;
    delete dataWord2;
    return;
  }

  if( lines != evtLgth){
    if ( verbose_ ) edm::LogWarning("dttf_unpacker")
      << "Number of words read != event length " << dec << lines << " " << evtLgth;
    delete dataWord1;
    delete dataWord2;
    return;
  }


  // --> analyse event    

  for( DTTFiterator =  DTTFWordContainer.begin();
       DTTFiterator != DTTFWordContainer.end();
       DTTFiterator++ ){

    DTTFChan = ((*DTTFiterator)&0xFF000000)>>24;
    DTTFiterator++;
    bitsID   = ((*DTTFiterator)&0xF0000000)>>28;

    int bxID     = bxNr(DTTFChan);
    if(bxID     == -999) continue;
    int wheelID  = wheel(DTTFChan);    
    if(wheelID  == -999) continue;
    int sectorID = sector(DTTFChan);    
    if(sectorID == -999) continue;

    //Input
    if(wheelID!=0 && bitsID<=0x9){   

      int wheelPh   = (abs(wheelID)-1)*wheelID/abs(wheelID); 
      int stationID = 0;
      int     ra    = 0;
      int     ba    = 0;
      int tsqual    = 0;
      int ts2tag    = 0;

      if ( ( bitsID >> 1 ) == 0 ){ stationID = 1;}
      if ( ( bitsID >> 1 ) == 1 ){ stationID = 2;}
      if ( ( bitsID >> 1 ) == 4 ){ stationID = 3;}
      if ( ( bitsID >> 1 ) == 2 ){ stationID = 4;}

      if(stationID != 3){ 
	  
        ts2tag = (bitsID)&0x1;
	tsqual = (~(*DTTFiterator)&0x07)-1;
	ba     = (~(*DTTFiterator)&0x1FF8)>>3;
        if( ba>0x1FF) ba-=0x400;
	ra     = (~(*DTTFiterator)&0x1FFE000)>>13;
        if( ra>0x7FF) ra-=0x1000;
      }   
      else{ 

        ts2tag = (bitsID)&0x1;
	tsqual = (~(*DTTFiterator)&0x07)-1;
	ra     = (~(*DTTFiterator)&0x7FF8)>>3;
        if( ra>0x7FF) ra-=0x1000;
      }

      if(tsqual!=7 && wheelID!=-1){
        phiSegments.push_back(
		    L1MuDTChambPhDigi( bxID+ts2tag, wheelPh, sectorID, stationID,
		    ra, ba, tsqual, ts2tag, 0) );
      }
    }
    //Input

    //Input
    if(wheelID==0 && bitsID<=0x4){
   
      int wheelTh = bitsID-2;

      int posALL, posBTI[7];

      if ( wheelTh == -2 || wheelTh == -1 || 
	   ( wheelTh == 0 && (sectorID == 0 || sectorID == 3 || sectorID == 4 || sectorID == 7 || sectorID == 8 || sectorID == 11) ) ) {

	posALL    =  ~(*DTTFiterator)&0x7F;
	posBTI[0] =  ~(*DTTFiterator)&0x01;
	posBTI[1] = (~(*DTTFiterator)&0x02)>>1;
	posBTI[2] = (~(*DTTFiterator)&0x04)>>2;
	posBTI[3] = (~(*DTTFiterator)&0x08)>>3;
	posBTI[4] = (~(*DTTFiterator)&0x10)>>4;
	posBTI[5] = (~(*DTTFiterator)&0x20)>>5;
	posBTI[6] = (~(*DTTFiterator)&0x40)>>6;

	if(posALL){
	  theSegments.push_back(
		      L1MuDTChambThDigi( bxID, wheelTh, sectorID, 1, posBTI) );
	}

	posALL    =  ~(*DTTFiterator)&0x3F80;
	posBTI[0] = (~(*DTTFiterator)&0x0080)>>7;
	posBTI[1] = (~(*DTTFiterator)&0x0100)>>8;
	posBTI[2] = (~(*DTTFiterator)&0x0200)>>9;
	posBTI[3] = (~(*DTTFiterator)&0x0400)>>10;
	posBTI[4] = (~(*DTTFiterator)&0x0800)>>11;
	posBTI[5] = (~(*DTTFiterator)&0x1000)>>12;
	posBTI[6] = (~(*DTTFiterator)&0x2000)>>13;

	if(posALL){
	  theSegments.push_back(
		      L1MuDTChambThDigi( bxID, wheelTh, sectorID, 2, posBTI) );
	}

	posALL    =  ~(*DTTFiterator)&0x1FC000;
	posBTI[0] = (~(*DTTFiterator)&0x004000)>>14;
	posBTI[1] = (~(*DTTFiterator)&0x008000)>>15;
	posBTI[2] = (~(*DTTFiterator)&0x010000)>>16;
	posBTI[3] = (~(*DTTFiterator)&0x020000)>>17;
	posBTI[4] = (~(*DTTFiterator)&0x040000)>>18;
	posBTI[5] = (~(*DTTFiterator)&0x080000)>>19;
	posBTI[6] = (~(*DTTFiterator)&0x100000)>>20;

	if(posALL){
	  theSegments.push_back(
		      L1MuDTChambThDigi( bxID, wheelTh, sectorID, 3, posBTI) );
	}
      }

      else {

	posALL    =  ~(*DTTFiterator)&0x7F;
	posBTI[6] =  ~(*DTTFiterator)&0x01;
	posBTI[5] = (~(*DTTFiterator)&0x02)>>1;
	posBTI[4] = (~(*DTTFiterator)&0x04)>>2;
	posBTI[3] = (~(*DTTFiterator)&0x08)>>3;
	posBTI[2] = (~(*DTTFiterator)&0x10)>>4;
	posBTI[1] = (~(*DTTFiterator)&0x20)>>5;
	posBTI[0] = (~(*DTTFiterator)&0x40)>>6;

	if(posALL){
	  theSegments.push_back(
		      L1MuDTChambThDigi( bxID, wheelTh, sectorID, 1, posBTI) );
	}

	posALL    =  ~(*DTTFiterator)&0x3F80;
	posBTI[6] = (~(*DTTFiterator)&0x0080)>>7;
	posBTI[5] = (~(*DTTFiterator)&0x0100)>>8;
	posBTI[4] = (~(*DTTFiterator)&0x0200)>>9;
	posBTI[3] = (~(*DTTFiterator)&0x0400)>>10;
	posBTI[2] = (~(*DTTFiterator)&0x0800)>>11;
	posBTI[1] = (~(*DTTFiterator)&0x1000)>>12;
	posBTI[0] = (~(*DTTFiterator)&0x2000)>>13;

	if(posALL){
	  theSegments.push_back(
		      L1MuDTChambThDigi( bxID, wheelTh, sectorID, 2, posBTI) );
	}

	posALL    =  ~(*DTTFiterator)&0x1FC000;
	posBTI[6] = (~(*DTTFiterator)&0x004000)>>14;
	posBTI[5] = (~(*DTTFiterator)&0x008000)>>15;
	posBTI[4] = (~(*DTTFiterator)&0x010000)>>16;
	posBTI[3] = (~(*DTTFiterator)&0x020000)>>17;
	posBTI[2] = (~(*DTTFiterator)&0x040000)>>18;
	posBTI[1] = (~(*DTTFiterator)&0x080000)>>19;
	posBTI[0] = (~(*DTTFiterator)&0x100000)>>20;

	if(posALL){
	  theSegments.push_back(
		      L1MuDTChambThDigi( bxID, wheelTh, sectorID, 3, posBTI) );
	}
      }
    }
    //Input

    //Addresses
    if(wheelID!=0 && bitsID>=0xA && bitsID<=0xB){   

      int candID   = bitsID - 0xA;

      addr4[candID]   = ((*DTTFiterator)&0x0F);
      addr3[candID]   = ((*DTTFiterator)&0xF0)>>4;
      addr2[candID]   = ((*DTTFiterator)&0xF00)>>8;
      addr1[candID]   = ((*DTTFiterator)&0x3000)>>12;
    }
    //Addresses

    //Output
    if(wheelID!=0 && bitsID>=0xC){   

      int muonID = 0;
      int     pt = 0;
      int     ch = 0;
      int    phi = 0;
      int   qual = 0;  

      muonID = (bitsID&0x1);
      qual   = (~(*DTTFiterator)&0x07);
      phi    = ((*DTTFiterator)&0x7F8)>>3;
      ch     = (~(*DTTFiterator)&0x800)>>11;
      pt     = (~(*DTTFiterator)&0x1F000)>>12;

      if(qual!=0){
        dtTracks.push_back(
		    L1MuDTTrackCand( 0, phi, 0, pt, ch, 1, 0, qual, bxID, wheelID, sectorID,
                    muonID, addr1[muonID], addr2[muonID], addr3[muonID], addr4[muonID] ) );
      }
    }
    //Output

    //Output
    if(wheelID==0 && bitsID>=0x8){
   
      int wheelTh = bitsID&0x7;

      int etaALL;

      etaALL =  ~(*DTTFiterator)&0x007F;
      if (etaALL) {
        etTrack[bxID+1][sectorID][wheelTh][0] =   (*DTTFiterator)&0x003F;
        efTrack[bxID+1][sectorID][wheelTh][0] = (~(*DTTFiterator)&0x0040)>>6;
      }

      etaALL =  (~(*DTTFiterator)&0x3F80)>>7;
      if (etaALL) {
        etTrack[bxID+1][sectorID][wheelTh][1]= ( (*DTTFiterator)&0x1F80)>>7;
        efTrack[bxID+1][sectorID][wheelTh][1]= (~(*DTTFiterator)&0x2000)>>13;
      }
    }
    //Output

  } // end for-loop container content

  delete dataWord1;
  delete dataWord2;
  return;
}

void DTTFFEDReader::match() {

  for ( L1MuDTTrackContainer::TrackIterator i  = dtTracks.begin();
                                            i != dtTracks.end();
                                            i++ ) {
    int bxTh     = i->bx()+1;
    int sectorTh = i->scNum();
    int wheelTh  = i->whNum()+3;
    if(wheelTh > 3) wheelTh-=1;
    int muonTh   = i->TrkTag();

    i->setEtaPacked(etTrack[bxTh][sectorTh][wheelTh][muonTh]);
    i->setFineHaloPacked(efTrack[bxTh][sectorTh][wheelTh][muonTh]);
  }

  return;
}

// access data
const L1MuDTChambPhContainer::Phi_Container& DTTFFEDReader::p_data() {
  return phiSegments;
}

const L1MuDTChambThContainer::The_Container& DTTFFEDReader::t_data() {
  return theSegments;
}

const L1MuDTTrackContainer::TrackContainer&  DTTFFEDReader::k_data() {
  return dtTracks;
}

void DTTFFEDReader::clear() {
  phiSegments.clear();
  theSegments.clear();
  dtTracks.clear();

  for(int i=0; i<3;  i++){
    for(int j=0; j<12; j++){
      for(int k=0; k<6;  k++){
        for(int l=0; l<2;  l++){
          etTrack[i][j][k][l] = 0;
          efTrack[i][j][k][l] = 0;
        } 
      } 
    } 
  }

  return;
}

int DTTFFEDReader::channel( int wheel, int sector,  int bx ){

  // wheel  :  -3 -2 -1 +1 +2 +3 <=> PHTF's : N2, N1, N0, P0, P1, P2
  //                           0 <=> ETTF  
  // sector :  0 -> 11
  // bx     : -1 -> +1

  int myChannel = 255;

  if ( abs(bx) > 1)               { return myChannel; }
  if ( sector < 0 || sector > 11) { return myChannel; }
  if ( abs(wheel) > 3)            { return myChannel; }

  myChannel = sector*21 + wheel*3 - bx + 10 ; 

  if (myChannel > 125) myChannel += 2;

  return myChannel;
}

int DTTFFEDReader::bxNr( int channel ){

  int myChannel = channel;

  if (myChannel > 127) myChannel -= 2;

  if (myChannel < 0 || myChannel > 251 ){ return -999; }

  int myBx = 1-(myChannel%3);

  return myBx;
}

int DTTFFEDReader::sector( int channel ){

  int myChannel = channel;

  if (myChannel > 127) myChannel -= 2;

  if (myChannel < 0 || myChannel > 251 ){ return -999; }

  return myChannel/21;
}

int DTTFFEDReader::wheel( int channel ){

  int myChannel = channel;

  if (myChannel > 127) myChannel -= 2;

  if (myChannel < 0 || myChannel > 251 ){ return -999; }

  int myWheel = ((myChannel%21)/3)-3;

  return myWheel;
}

void DTTFFEDReader::calcCRC(int myD1, int myD2, int &myC){

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
	       D[33] + D[32] + D[20] + D[18] + D[5]  + D[4]  +
	       C[2]  + C[12] + C[13] )%2;

  myCRC[7] = ( D[62] + D[61] + D[51] + D[47] + D[36] + D[35] +
	       D[34] + D[33] + D[21] + D[19] + D[6]  + D[5]  +
	       C[3]  + C[13] + C[14] )%2;

  myCRC[8] = ( D[63] + D[62] + D[52] + D[48] + D[37] + D[36] +
	       D[35] + D[34] + D[22] + D[20] + D[7]  + D[6]  +
	       C[0]  + C[4]  + C[14] + C[15] )%2;

  myCRC[9] = ( D[63] + D[53] + D[49] + D[38] + D[37] + D[36] +
	       D[35] + D[23] + D[21] + D[8]  + D[7]  + C[1]  +
	       C[5]  + C[15] )%2;

  myCRC[10] = ( D[54] + D[50] + D[39] + D[38] + D[37] + D[36] + 
		D[24] + D[22] + D[9]  + D[8]  + C[2]  + C[6] )%2;

  myCRC[11] = ( D[55] + D[51] + D[40] + D[39] + D[38] + D[37] +
		D[25] + D[23] + D[10] + D[9]  + C[3]  + C[7] )%2;

  myCRC[12] = ( D[56] + D[52] + D[41] + D[40] + D[39] + D[38] +
		D[26] + D[24] + D[11] + D[10] + C[4]  + C[8] )%2;

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
