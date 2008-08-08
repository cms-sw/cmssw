#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData2007.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string.h> // for bzero


CSCAnodeData2007::CSCAnodeData2007(const CSCALCTHeader & header)
  : nAFEBs_(header.nLCTChipRead()), nTimeBins_(header.NTBins())
{
  bzero(theDataFrames, sizeInWords()*2);
  init(header);
}


CSCAnodeData2007::CSCAnodeData2007(const CSCALCTHeader & header ,
                                   const unsigned short *buf) 
  : nAFEBs_(header.nLCTChipRead()), nTimeBins_(header.NTBins())
{
    init(header);
    memcpy(theDataFrames, buf, sizeInWords()*2);///dont memcpy if not 2006 or 2007
}


void CSCAnodeData2007::init(const CSCALCTHeader & header) {
  ///the sizes of raw words vary depending on type of the ALCT board
  ///                         number of layer parts for various
  ///                         alct board types:     1  2  3     5  6
  static unsigned short int layerParts[7]    = { 3, 3, 4, 6, 6, 8,10};
  static unsigned short int wireGroups[7]    = {32,32,48,64,64,96,112};

  sizeInWords2007_=(1-header.alctHeader2007().rawOverflow)*6*
  header.alctHeader2007().rawBins*layerParts[header.alctHeader2007().boardType];
  layerParts_ = layerParts[header.alctHeader2007().boardType];
  maxWireGroups_ = wireGroups[header.alctHeader2007().boardType];
}


std::vector<CSCWireDigi> CSCAnodeData2007::wireDigis(int layer) const {
  std::vector<CSCWireDigi> digis;
  uint32_t tbinbits=0;
  uint16_t wireGroup=0;
    for(int layerPart = 0; layerPart <layerParts_; ++layerPart) {
      ///we know how many layer parts are there from ALCT header 
      for (int j=0; (j<12)&&((layerPart*12+j)<maxWireGroups_) ;++j) {
	///loop over 12 bits in each word (each bit is one wiregroup) 
	///we want to stop if we reached the maxWireGroups
	for(int tbin = 0; tbin < nTimeBins_; ++tbin) { ///loop over tbins
	  const CSCAnodeDataFrame2007 & frame = findFrame(tbin, layer, layerPart);
	  if(frame.data() != 0) {
	    if(frame.isHit(j)) {
	      tbinbits=tbinbits + (1<<tbin);
	    }
	  }
	}//end of tbin loop
	if (tbinbits !=0 ) {
	  wireGroup = (layerPart*12+j)+1;
	  CSCWireDigi digi(wireGroup, tbinbits);
	    LogTrace ("CSCAnodeData|CSCRawToDigi") << "Layer " << layer << " " << digi;
	  digis.push_back(digi);
	  tbinbits=0;
	}
      }///end of the loop over bits in the data frame
    }///end of the loop over layer parts
    
  return digis;
}


const CSCAnodeDataFrame2007 & CSCAnodeData2007::findFrame(int tbin, int layer, int layerPart) const {
  return (const CSCAnodeDataFrame2007 &)(theDataFrames[index(tbin, layer, layerPart)]);
}


CSCAnodeDataFrame2007 & CSCAnodeData2007::findFrame(int tbin, int layer, int layerPart) {
  return (CSCAnodeDataFrame2007 &)(theDataFrames[index(tbin, layer, layerPart)]);
}


int CSCAnodeData2007::index(int tbin, int layer, int layerPart) const 
{
  assert(tbin<nTimeBins_);
  assert(layer<=6);
  assert(layerPart<layerParts_);
  return tbin*6*layerParts_+(layer-1)*layerParts_+layerPart;
}


void CSCAnodeData2007::add(const CSCWireDigi & digi, int layer) 
{
/*
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
      rawHit(alctBoard, bxn, layer, halfLayer).addHit(bitNumber);
    } 
  else 
    {
      LogTrace("CSCAnodeData|CSCRawToDigi")<< "warning: not saving anode data in bx " << bxn 
			      << ": out of range ";
    }
*/
}

