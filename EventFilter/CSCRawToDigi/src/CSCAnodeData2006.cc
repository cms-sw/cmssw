#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData2006.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string.h> // for bzero

bool debug = false;
#include <iostream>

CSCAnodeDataFrame2006::CSCAnodeDataFrame2006(unsigned chip, unsigned tbin, unsigned data)
: theFrame(0)
{
   // lowest bit, plus the OR of the next two.
   unsigned packedChip = ( (chip&1) + 2*(chip>1) );
   theFrame = data + ((tbin&0x1F) << 8) + (packedChip<<13);
}


CSCAnodeData2006::CSCAnodeData2006(const CSCALCTHeader & header) ///for digi->raw packing
  : nAFEBs_(header.nLCTChipRead()), nTimeBins_(header.NTBins()) 
{
  LogTrace ("CSCAnodeData|CSCRawToDigi") << "Making Anode data " 
			      << sizeInWords() << " AFEB " << nAFEBs_ 
			      << " TBINS " << nTimeBins_;
  bzero(theDataFrames, sizeInWords()*2);
  for(int afeb = 0; afeb < nAFEBs_; ++afeb) {
    for(int tbin = 0; tbin < nTimeBins_; ++tbin) {
      for(int layer = 1; layer <= 6; ++layer) {
	for(int halfLayer = 0; halfLayer < 2; ++halfLayer) {
	  theDataFrames[index(afeb, tbin, layer)+halfLayer] = CSCAnodeDataFrame2006(afeb, tbin, 0).frame();
	}
      }
    }
  }
    /// To get BX from ALCT digis
    alctBX_= header.BXNCount();
}

// initialize
CSCAnodeData2006::CSCAnodeData2006(const CSCALCTHeader & header ,
                               const unsigned short *buf) 
  : nAFEBs_(header.nLCTChipRead()), 
    nTimeBins_(header.NTBins())
{

  ///the sizes of raw words vary depending on type of the ALCT board
  ///                         number of layer parts for various
  ///                         alct board types:     1  2  3     5  6
  LogTrace ("CSCAnodeData|CSCRawToDigi") << "nAFEBs = " << nAFEBs_ << "  nTimeBins = " 
				<< nTimeBins_ << " nFrames = " << sizeInWords();  
  LogTrace ("CSCAnodeData|CSCRawToDigi") << header << " HEADER CHECK " << header.check();

  memcpy(theDataFrames, buf, sizeInWords()*2);///dont memcpy if not 2006 or 2007
}


std::vector<CSCWireDigi> CSCAnodeData2006::wireDigis(int layer) const {
  std::vector<CSCWireDigi> digis;
  uint32_t tbinbits=0;
  uint16_t wireGroup=0;
    for(int afeb = 0; afeb < nAFEBs_; ++afeb) {
      for(int halfLayer = 0; halfLayer <2; ++halfLayer) {
	for (int j=0;j<8;++j) {
	  for(int tbin = 0; tbin < nTimeBins_; ++tbin) {
	    CSCAnodeDataFrame2006 frame(rawHit(afeb,tbin,layer, halfLayer));
	    // see if there's anything in 1st 8 bits.  Usually zero
	    if(frame.data() != 0) {
	      if(frame.isHit(j)) {
		tbinbits=tbinbits + (1<<tbin);      
	      }
	    }
	  }//end of tbin loop
	  if (tbinbits !=0 ) {
	    wireGroup = (afeb*16+halfLayer*8+j)+1;
            uint32_t wireGroupBX=alctBX_;
            wireGroup = wireGroup | (wireGroupBX << 16);
	    CSCWireDigi digi(wireGroup, tbinbits);
	    if (debug)
	      LogTrace ("CSCAnodeData|CSCRawToDigi") << "Layer " << layer << " " << digi;
	    digis.push_back(digi);
	    tbinbits=0;
	  }
	}
      }
    }
    
  return digis;
}


void CSCAnodeData2006::add(const CSCWireDigi & digi, int layer) 
{

  int wireGroup = digi.getWireGroup();
  int bxn=digi.getBeamCrossingTag(); 
  int alctBoard  = (wireGroup-1) / 16;
  int localGroup = (wireGroup-1) % 16;

  // crash if there's a bad wire number, but don't freak out
  // if a time bin is out of range 
  //  assert(alctBoard < nAFEBs_);
  if(alctBoard > nAFEBs_)
    {
      edm::LogError("CSCAnodeData|CSCRawToDigi") << "Bad Wire Number for this digi.";
      return;
    }
  if(bxn >= 0 && bxn < nTimeBins_) 
    {
      // 12 16-bit words per time bin, two per layer
      // wiregroups 0-7 go on the first line, 8-15 go on the 2nd.
      unsigned halfLayer = (localGroup > 7);
      unsigned bitNumber = localGroup % 8;
      // and pack it in the 8 bits allocated
      addHit(alctBoard, bxn, layer, halfLayer, bitNumber);
    } 
  else 
    {
      LogTrace("CSCAnodeData|CSCRawToDigi")<< "warning: not saving anode data in bx " << bxn 
			      << ": out of range ";
    }
}


void CSCAnodeData2006::addHit(int afeb, int tbin, int layer, int halfLayer, unsigned wireBit)
{
  int i = index(afeb,tbin,layer) + halfLayer;
  CSCAnodeDataFrame2006 frame(theDataFrames[i]);
  frame.addHit(wireBit);
  theDataFrames[i] = frame.frame(); 
}


CSCAnodeDataFrame2006 CSCAnodeData2006::rawHit(int afeb, int tbin, int layer, int halfLayer) const
{
  return CSCAnodeDataFrame2006(theDataFrames[index(afeb, tbin, layer)+halfLayer]);
}


int CSCAnodeData2006::index(int afeb, int tbin, int layer) const {
  int result = (layer-1)*2 + 12*tbin + afeb*12*nTimeBins_;
  assert(result < sizeInWords());
  return result;
}

#include <iostream>
void CSCAnodeData2006::selfTest()
{
  CSCAnodeDataFrame2006 frame(2, 15, 32); 
  assert(frame.chip() == 2);
  assert(frame.tbin() == 15);
  assert(frame.data() == 32);
  assert(frame.isHit(5));
  assert(!frame.isHit(7));
  frame.addHit(7);
  assert(frame.isHit(7));

  CSCWireDigi wireDigi(10, (1 << 4));
  CSCALCTHeader header(7);
  CSCAnodeData2006 anodeData(header);
  anodeData.add(wireDigi, 1);
  anodeData.add(wireDigi, 6);

  std::vector<CSCWireDigi> wires1 = anodeData.wireDigis(1);
  std::vector<CSCWireDigi> wires6 = anodeData.wireDigis(6);

  assert(wires1.size() == 1);
  assert(wires6.size() == 1);
  assert(wires1[0].getWireGroup() == 10);
  assert(wires6[0].getWireGroup() == 10);
}


