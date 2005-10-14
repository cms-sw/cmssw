#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include <iostream>
#include <string.h> // for bzero

bool CSCAnodeData::debug = false;

CSCAnodeData::CSCAnodeData() 
: nAFEBs_(0), nTimeBins_(0) {
  bzero(this, sizeInWords()*2);
}


CSCAnodeData::CSCAnodeData(const CSCALCTHeader & header)
: nAFEBs_(header.nLCTChipRead()), nTimeBins_(header.NTBins()) {
  if(debug) std::cout << header << std::endl;
  if(debug) std::cout << "Making Anode data " 
		      << nFrames() << " AFEB " << nAFEBs_ 
		      << " TBINS " << nTimeBins_ << std::endl;
  bzero(this, sizeInWords()*2);
  for(int afeb = 0; afeb < nAFEBs_; ++afeb) {
    for(int tbin = 0; tbin < nTimeBins_; ++tbin) {
      for(int layer = 1; layer <= 6; ++layer) {
        for(int halfLayer = 0; halfLayer < 2; ++halfLayer) {
          rawHit(afeb, tbin, layer, halfLayer)
            = CSCAnodeDataFrame(afeb, tbin, 0);
        }
      }
    }
  }
}


// initialize
CSCAnodeData::CSCAnodeData(const CSCALCTHeader & header ,
                               const unsigned short *buf) 
: nAFEBs_(header.nLCTChipRead()), nTimeBins_(header.NTBins()) {
  if (debug ) {
    std::cout << "nAFEBs = " << nAFEBs_ << "  nTimeBins = " << nTimeBins_
	      << " nFrames = " << nFrames() << std::endl;  
    std::cout << header << " HEADER CHECK " << header.check() << std::endl;
  }  
  memcpy(theDataFrames, buf, sizeInWords()*2);

  /// make sure the trailer code is there
  //cout << "Anode data " << endl;
  //if ( debug ) for(int i=0;i<sizeInWords();i++) cout << "+++ CSCAnodeData: "<< buf[i] << endl ;
  
  if (debug) {
    for (int i = 0; i < 4; ++i) {
      std::cout << std::hex << buf[i+nFrames()] << " ";
    }
    std::cout << std::dec << std::endl;
  }
}


/*
bool CSCAnodeData
int CSCAnodeData::unpackALCT(const unsigned short *buf) {
  int StartLine = 12;
  if (debug) cout << "unpackALCT() is called " << endl;

  // Determine endline (since this class is called we know it exists since nalct>0
  int i,j,k;
  int e0dLine=0;
  i = StartLine - 1;
  while (e0dLine==0) {
    i++;
    //if (i < 100) cout << "i " << i << " " << hex << buf[i] << dec << endl;
    if ((buf[i] & 0xfff) == 0xe0d) e0dLine = i;
    if ((buf[i] & 0xfff) == 0xe0b) {
      cout << "End of ALCT wasn't found... Skipping CSCAnodeData" << endl;    
      return -12; // to return buf pointer to the beginning of the event for CLCTData 
    }
  }
  int EndLine = e0dLine - 3;

  // *** Decodes Data ***

  if (EndLine > StartLine) {
    for (i=0;i<32;i++) for (j=0;j<6;j++) for (k=0;k<8;k++)
      {rawhit[i][j][k] = 0;}
    for (int counter=StartLine;counter<=EndLine;counter++) {
      Tbin = (buf[counter] >> 8) & 0x1f;
      OldLCTchip = LCTchip;
      LCTchip = (buf[counter] >> 13) & 3;
      if (LCTchip >= OldLCTchip) {
	if (Tbin > MaxTbin) MaxTbin = Tbin;
	if (debug && (buf[counter] & 0xff) > 0)
	  cout << " Tbin = " << Tbin << " Layer = " << Layer
	       << " ind2 = " << (Layer >> 1)
	       << " ind3 = " << ((LCTchip << 1) | (Layer & 1))
	       << " LCTchip = " << LCTchip
	       << " buffer = " << (buf[counter] & 0xff) << endl;
	rawhit[Tbin][Layer >> 1][(LCTchip << 1) | (Layer & 1)] = buf[counter] & 0xff;
	Layer++;
	if (Layer == 12) Layer = 0;
      }
    }

    fillDigis();
  }
  else {
    cout << "+++ CSCAnodeData warning: StartLine = " << StartLine
	 << " >= EndLine = " << EndLine << " +++" << endl;
  }

  return EndLine-StartLine+1;
}
*/

#ifndef UNPCK_ONLY

std::vector<CSCWireDigi> CSCAnodeData::wireDigis(int layer) const {
  std::vector<CSCWireDigi> digis;
  for(int afeb = 0; afeb < nAFEBs_; ++afeb) {
    for(int tbin = 0; tbin < nTimeBins_; ++tbin) {
      for(int halfLayer = 0; halfLayer <2; ++halfLayer) {
        const CSCAnodeDataFrame & frame = rawHit(afeb,tbin,layer, halfLayer);
        // see if there's anything in 1st 8 bits.  Usually zero
        if(frame.data() != 0) {
          for (int j=0;j<8;j++) {
            if(frame.isHit(j)) {
	      // wireGroup; 5th e time; FD time; beamCrossingTag; adcCounts
              int wireGroup = (afeb*16+halfLayer*8+j)+1;
              CSCWireDigi digi(wireGroup, 0, 0, tbin, 1);
              if (debug) std::cout << "Layer " << layer << " " << digi;
              digis.push_back(digi);
            }
          }
        }
      }
    }
  }
  return digis;
}

std::vector < std::vector<CSCWireDigi> > CSCAnodeData::wireDigis() const {
  std::vector < std::vector<CSCWireDigi> > result;
  for (int layer = 1; layer <= 6; ++layer) {
    result.push_back(wireDigis(layer));
  }
  return result;
}



void CSCAnodeData::add(const CSCWireDigi & digi, int layer) {

  int wireGroup = digi.getWireGroup();
  int bxn=digi.getBeamCrossingTag(); 
  int alctBoard  = (wireGroup-1) / 16;
  int localGroup = (wireGroup-1) % 16;

  // crash if there's a bad strip number, but don't freak out
  // if a time bin is out of range 
//  assert(alctBoard < nAFEBs_);
  if(alctBoard > nAFEBs_) {
    std::cerr << "Bad Wire Number for this digi.  Simulation wire geometry needs to be more realistic" << std::endl;
    std::cerr << "Skipping digi " << digi << std::endl;
    return;
  }

  if(bxn >= 0 && bxn < nTimeBins_) {
    // 12 16-bit words per time bin, two per layer
    // wiregroups 0-7 go on the first line, 8-15 go on the 2nd.
    unsigned halfLayer = (localGroup > 7);
    unsigned bitNumber = localGroup % 8;
    // and pack it in the 8 bits allocated
    rawHit(alctBoard, bxn, layer, halfLayer).addHit(bitNumber);
  } else {
    std::cout << "warning: not saving anode data in bx " << bxn 
         << ": out of range " << std::endl;
  }
}


bool CSCAnodeData::selfTest() {
  CSCALCTHeader dummyHeader(5);
  CSCAnodeData data(dummyHeader);
  CSCWireDigi oldDigi(50,0,0,4,1);
  data.add(oldDigi, 3);
  std::vector<CSCWireDigi> newDigis = data.wireDigis(3);
  std::cout << "digis size " << newDigis.size() << std::endl; 
  if(newDigis.size() > 0) {
    std::cout << "OLD DIGI " << oldDigi << std::endl;
    std::cout << "NEW DIGI " << newDigis[0] << std::endl;
  }
  return true;
}

#endif
