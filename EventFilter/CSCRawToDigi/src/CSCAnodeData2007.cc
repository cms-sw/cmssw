#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData2007.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string.h> // for bzero
#include<iostream>

CSCAnodeData2007::CSCAnodeData2007(const CSCALCTHeader & header)
  : nAFEBs_(header.nLCTChipRead()), nTimeBins_(header.NTBins())
{
  init(header);
  bzero(theDataFrames, sizeInWords()*2);
  /// To get BX from ALCT digis
  alctBX_= header.BXNCount();
}


CSCAnodeData2007::CSCAnodeData2007(const CSCALCTHeader & header ,
                                   const unsigned short *buf) 
  : nAFEBs_(header.nLCTChipRead()), nTimeBins_(header.NTBins())
{
    init(header);
    memcpy(theDataFrames, buf, sizeInWords()*2);///dont memcpy if not 2006 or 2007
    /// To get BX from ALCT digis
   alctBX_= header.BXNCount();
}


void CSCAnodeData2007::init(const CSCALCTHeader & header) {
  ///the sizes of raw words vary depending on type of the ALCT board
  ///                         number of layer parts for various
  ///                         alct board types:     1  2  3     5  6
  static unsigned short int layerParts[7]    = { 3, 3, 4, 6, 6, 8,10};
  static unsigned short int wireGroups[7]    = {32,32,48,64,64,96,112};
  //header.ALCTDigis();
  sizeInWords2007_=(1-header.alctHeader2007().rawOverflow)*6*
  header.alctHeader2007().rawBins*layerParts[header.alctHeader2007().boardType];
  layerParts_ = layerParts[header.alctHeader2007().boardType];
  maxWireGroups_ = wireGroups[header.alctHeader2007().boardType];
}


std::vector<CSCWireDigi> CSCAnodeData2007::wireDigis(int layer) const {
  std::vector<CSCWireDigi> digis;
  uint32_t tbinbits=0;
  uint32_t wireGroup=0;
  /// BX from ACT (first component)
  for(int layerPart = 0; layerPart <layerParts_; ++layerPart) {
    ///we know how many layer parts are there from ALCT header 
    for (int j=0; (j<12)&&((layerPart*12+j)<maxWireGroups_) ;++j) {
      ///loop over 12 bits in each word (each bit is one wiregroup) 
      ///we want to stop if we reached the maxWireGroups
      for(int tbin = 0; tbin < nTimeBins_; ++tbin) { ///loop over tbins
	CSCAnodeDataFrame2007 frame = findFrame(tbin, layer, layerPart);
	if(frame.data() != 0) {
	  if(frame.isHit(j)) {
	    tbinbits=tbinbits + (1<<tbin);
	  }
	}
      }//end of tbin loop
      if (tbinbits !=0 ) {
	wireGroup = (layerPart*12+j)+1;
	/// BX from ALCT encoded in wireGroup
	uint32_t wireGroupBX=alctBX_;
	wireGroup = wireGroup | (wireGroupBX << 16);
	CSCWireDigi digi(wireGroup, tbinbits);
	LogTrace ("CSCAnodeData|CSCRawToDigi") << "Layer " << layer << " " << digi;
	digis.push_back(digi);
	tbinbits=0;
      }
    }///end of the loop over bits in the data frame
  }///end of the loop over layer parts
  
  return digis;
}


CSCAnodeDataFrame2007 CSCAnodeData2007::findFrame(int tbin, int layer, int layerPart) const {
  return CSCAnodeDataFrame2007(theDataFrames[index(tbin, layer, layerPart)]);
}


int CSCAnodeData2007::index(int tbin, int layer, int layerPart) const 
{
  assert(tbin<nTimeBins_);
  assert(layer<=6);
  assert(layerPart<layerParts_);
  int result = tbin*6*layerParts_+(layer-1)*layerParts_+layerPart;
  assert(result < MAXFRAMES);
  return result;
}


void CSCAnodeData2007::add(const CSCWireDigi & digi, int layer) 
{
  int wireGroup = digi.getWireGroup();
  //           wireGroup = (layerPart*12+j)+1;
  unsigned layerPart = (wireGroup-1) / 12;
  unsigned wireInPart = (wireGroup-1) % 12; 

  std::vector<int> timeBinsOn = digi.getTimeBinsOn();
  for(std::vector<int>::const_iterator timeBinOn = timeBinsOn.begin();
      timeBinOn != timeBinsOn.end(); ++timeBinOn)
  {
    // crash if there's a bad wire number, but don't freak out
    // if a time bin is out of range 
    //  assert(alctBoard < nAFEBs_);
    if(layerPart >= layerParts_)
      {
        edm::LogError("CSCAnodeData|CSCRawToDigi") << "Bad Wire Number for this digi.";
        return;
      }

    if((*timeBinOn) >= 0 && (*timeBinOn) < nTimeBins_) 
      {
        CSCAnodeDataFrame2007 frame = findFrame(*timeBinOn, layer, layerPart);
        frame.addHit(wireInPart);
        // FIXME doesn't carry over the (currently 0) leading bits
        theDataFrames[index(*timeBinOn, layer, layerPart)]  = frame.data();
      } 
    else 
      {
        LogTrace("CSCAnodeData|CSCRawToDigi")<< "warning: not saving anode data in bx " << *timeBinOn 
  			      << ": out of range ";
      }
  }
}

void CSCAnodeData2007::selfTest()
{
  int wireGroup = 12;
  int timeBin = 6;
  CSCWireDigi wireDigi(wireGroup, (1 << timeBin));
  CSCALCTHeader header(1);
  CSCAnodeData2007 anodeData(header);
  anodeData.add(wireDigi, 1);
  anodeData.add(wireDigi, 6);

  std::vector<CSCWireDigi> wires1 = anodeData.wireDigis(1);
  std::vector<CSCWireDigi> wires6 = anodeData.wireDigis(6);

  assert(wires1.size() == 1);
  assert(wires6.size() == 1);
  assert(wires1[0].getWireGroup() == wireGroup);
  assert(wires6[0].getWireGroup() == wireGroup);
  assert(wires1[0].getTimeBin() == timeBin);
  assert(wires6[0].getTimeBin() == timeBin);

}

