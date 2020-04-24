#ifndef RctDataDecoder_hh
#define RctDataDecoder_hh

#include "EventFilter/RctRawToDigi/src/RCTInfo.hh"

/*
 * Extract RCT Object Info from oRSC Output Fibers
 * Primarily intended for use on CTP7/MP7
 */

class RctDataDecoder{
public:
  
  bool decodeLinks(const std::vector <unsigned int> evenFiberData, 
		   const std::vector <unsigned int> oddFiberData,
		   std::vector <RCTInfo> &rctInfoData) {
  // Ensure that there is data to process
  unsigned int nWordsToProcess = evenFiberData.size();
  unsigned int remainder = nWordsToProcess%6;
  if(nWordsToProcess != oddFiberData.size()) {
    //std::cerr << "RCTInfoFactory::produce -- even and odd fiber sizes are different!" << std::endl;
    return false;
  }
  if(nWordsToProcess == 0|| nWordsToProcess/6 == 0) {
    //std::cerr << "RCTInfoFactory::produce -- evenFiberData is null :(" << std::endl;
    return false;
  }
  else if((nWordsToProcess % 6) != 0) {
    nWordsToProcess=nWordsToProcess-remainder;
  }

  // Extract RCTInfo

  unsigned int nBXToProcess = nWordsToProcess / 6;

  for(unsigned int iBX = 0; iBX < nBXToProcess; iBX++) {

    RCTInfo rctInfo;
    // We extract into rctInfo the data from RCT crate
    // Bit field description can be found in the spreadsheet:
    // https://twiki.cern.ch/twiki/pub/CMS/ORSCOperations/oRSCFiberDataSpecificationV5.xlsx
    // Even fiber bits contain 4x4 region information
    rctInfo.rgnEt[0][0]  = (evenFiberData[iBX * 6 + 0] & 0x0003FF00) >>  8;
    rctInfo.rgnEt[0][1]  = (evenFiberData[iBX * 6 + 0] & 0x0FFC0000) >> 18;
    rctInfo.rgnEt[1][0]  = (evenFiberData[iBX * 6 + 0] & 0xF0000000) >> 28;
    rctInfo.rgnEt[1][0] |= (evenFiberData[iBX * 6 + 1] & 0x0000003F) <<  4;
    rctInfo.rgnEt[1][1]  = (evenFiberData[iBX * 6 + 1] & 0x0000FFC0) >>  6;
    rctInfo.rgnEt[2][0]  = (evenFiberData[iBX * 6 + 1] & 0x03FF0000) >> 16;
    rctInfo.rgnEt[2][1]  = (evenFiberData[iBX * 6 + 1] & 0xFC000000) >> 26;
    rctInfo.rgnEt[2][1] |= (evenFiberData[iBX * 6 + 2] & 0x0000000F) <<  6;
    rctInfo.rgnEt[3][0]  = (evenFiberData[iBX * 6 + 2] & 0x00003FF0) >>  4;
    rctInfo.rgnEt[3][1]  = (evenFiberData[iBX * 6 + 2] & 0x00FFC000) >> 14;
    rctInfo.rgnEt[4][0]  = (evenFiberData[iBX * 6 + 2] & 0xFF000000) >> 24;
    rctInfo.rgnEt[4][0] |= (evenFiberData[iBX * 6 + 3] & 0x00000003) <<  8;
    rctInfo.rgnEt[4][1]  = (evenFiberData[iBX * 6 + 3] & 0x00000FFC) >>  2;
    rctInfo.rgnEt[5][0]  = (evenFiberData[iBX * 6 + 3] & 0x003FF000) >> 12;
    rctInfo.rgnEt[5][1]  = (evenFiberData[iBX * 6 + 3] & 0xFFC00000) >> 22;
    rctInfo.rgnEt[6][0]  = (evenFiberData[iBX * 6 + 4] & 0x000003FF) >>  0;
    rctInfo.rgnEt[6][1]  = (evenFiberData[iBX * 6 + 4] & 0x000FFC00) >> 10;
    rctInfo.tBits  = (evenFiberData[iBX * 6 + 4] & 0xFFF00000) >> 20;
    rctInfo.tBits |= (evenFiberData[iBX * 6 + 5] & 0x00000003) << 12; //bug? 4 to 5
    rctInfo.oBits  = (evenFiberData[iBX * 6 + 5] & 0x0000FFFC) >>  2;
    rctInfo.c4BC0  = (evenFiberData[iBX * 6 + 5] & 0x000C0000) >> 18;
    rctInfo.c5BC0  = (evenFiberData[iBX * 6 + 5] & 0x00300000) >> 20;
    rctInfo.c6BC0  = (evenFiberData[iBX * 6 + 5] & 0x00C00000) >> 22;
    // Odd fiber bits contain 2x1, HF and other miscellaneous information
    rctInfo.hfEt[0][0]  = (oddFiberData[iBX * 6 + 0] & 0x0000FF00) >>  8;
    rctInfo.hfEt[0][1]  = (oddFiberData[iBX * 6 + 0] & 0x00FF0000) >> 16;
    rctInfo.hfEt[1][0]  = (oddFiberData[iBX * 6 + 0] & 0xFF000000) >> 24;
    rctInfo.hfEt[1][1]  = (oddFiberData[iBX * 6 + 1] & 0x000000FF) >>  0;
    rctInfo.hfEt[0][2]  = (oddFiberData[iBX * 6 + 1] & 0x0000FF00) >>  8;
    rctInfo.hfEt[0][3]  = (oddFiberData[iBX * 6 + 1] & 0x00FF0000) >> 16;
    rctInfo.hfEt[1][2]  = (oddFiberData[iBX * 6 + 1] & 0xFF000000) >> 24;
    rctInfo.hfEt[1][3]  = (oddFiberData[iBX * 6 + 2] & 0x000000FF) >>  0;
    rctInfo.hfQBits     = (oddFiberData[iBX * 6 + 2] & 0x0000FF00) >>  8;
    rctInfo.ieRank[0]   = (oddFiberData[iBX * 6 + 2] & 0x003F0000) >> 16;
    rctInfo.ieRegn[0]   = (oddFiberData[iBX * 6 + 2] & 0x00400000) >> 22;
    rctInfo.ieCard[0]   = (oddFiberData[iBX * 6 + 2] & 0x03800000) >> 23; //bug? 25 to 23
    rctInfo.ieRank[1]   = (oddFiberData[iBX * 6 + 2] & 0xFC000000) >> 26;
    rctInfo.ieRegn[1]   = (oddFiberData[iBX * 6 + 3] & 0x00000001) >>  0;
    rctInfo.ieCard[1]   = (oddFiberData[iBX * 6 + 3] & 0x0000000E) >>  1;
    rctInfo.ieRank[2]   = (oddFiberData[iBX * 6 + 3] & 0x000003F0) >>  4;
    rctInfo.ieRegn[2]   = (oddFiberData[iBX * 6 + 3] & 0x00000400) >> 10;
    rctInfo.ieCard[2]   = (oddFiberData[iBX * 6 + 3] & 0x00003800) >> 11;
    rctInfo.ieRank[3]   = (oddFiberData[iBX * 6 + 3] & 0x000FC000) >> 14;
    rctInfo.ieRegn[3]   = (oddFiberData[iBX * 6 + 3] & 0x00100000) >> 20;
    rctInfo.ieCard[3]   = (oddFiberData[iBX * 6 + 3] & 0x00E00000) >> 21;
    rctInfo.neRank[0]   = (oddFiberData[iBX * 6 + 3] & 0x3F000000) >> 24; 
    rctInfo.neRegn[0]   = (oddFiberData[iBX * 6 + 3] & 0x40000000) >> 30;
    rctInfo.neCard[0]   = (oddFiberData[iBX * 6 + 3] & 0x80000000) >> 31; 
    rctInfo.neCard[0]  |= (oddFiberData[iBX * 6 + 4] & 0x00000003) <<  1; //bug? >> 0 to << 1
    rctInfo.neRank[1]   = (oddFiberData[iBX * 6 + 4] & 0x000000FC) >>  2;
    rctInfo.neRegn[1]   = (oddFiberData[iBX * 6 + 4] & 0x00000100) >>  8;
    rctInfo.neCard[1]   = (oddFiberData[iBX * 6 + 4] & 0x00000E00) >>  9;
    rctInfo.neRank[2]   = (oddFiberData[iBX * 6 + 4] & 0x0003F000) >> 12;
    rctInfo.neRegn[2]   = (oddFiberData[iBX * 6 + 4] & 0x00040000) >> 18;
    rctInfo.neCard[2]   = (oddFiberData[iBX * 6 + 4] & 0x00380000) >> 19;
    rctInfo.neRank[3]   = (oddFiberData[iBX * 6 + 4] & 0x0FC00000) >> 22;
    rctInfo.neRegn[3]   = (oddFiberData[iBX * 6 + 4] & 0x10000000) >> 28;
    rctInfo.neCard[3]   = (oddFiberData[iBX * 6 + 4] & 0xE0000000) >> 29;
    rctInfo.mBits       = (oddFiberData[iBX * 6 + 5] & 0x00003FFF) >>  0;
    rctInfo.c1BC0       = (oddFiberData[iBX * 6 + 5] & 0x00030000) >> 16;
    rctInfo.c2BC0       = (oddFiberData[iBX * 6 + 5] & 0x000C0000) >> 18;
    rctInfo.c3BC0       = (oddFiberData[iBX * 6 + 5] & 0x00300000) >> 20;
    rctInfoData.push_back(rctInfo);
  }
  return true;
  
  }

  bool setRCTInfoCrateID(std::vector<RCTInfo> &rctInfoVector, unsigned int crateID)
  {
    for( unsigned int i = 0 ; i < rctInfoVector.size() ; i++ ){
      rctInfoVector.at(i).crateID = crateID;
    }
    
    return true;

  }

  bool decodeLinkID(const uint32_t inputValue, uint32_t &crateNumber, bool &even)
  {
    uint32_t linkNumber;
    //if crateNumber not valid set to 0xFF
    crateNumber = ( inputValue >> 8 ) & 0xFF;
    if(crateNumber > 17)
      crateNumber = 0xFF;
    
    //if linkNumber not valid set to 0xFF
    linkNumber = ( inputValue ) & 0xFF;
    
    if(linkNumber > 12)
      linkNumber = 0xFF;
    
    if((linkNumber&0x1) == 0)
      even=true;
    else
      even=false;
    
    return true;
  };

  bool getExpectedLinkID(const uint32_t iLink, uint32_t &crateID, bool &even)
  {
    if(iLink == 0)//: 90a 
      {crateID = 9; even = true;}
    else if(iLink == 1) //a0a
      {crateID = 10; even = true;}
    else if(iLink == 2)// 60a
      {crateID = 6; even = true;}
    else if(iLink == 3)// 90b
      {crateID = 9; even = false;}
    else if(iLink == 4)// 60b
      {crateID = 6; even = false;}
    else if(iLink == 5)// 706
      {crateID = 7; even = true;}
    else if(iLink == 6)// b0a
      {crateID = 11; even = true;}
    else if(iLink == 7)// 70b
      {crateID = 7; even = false;}
    else if(iLink == 8)// 80b
      {crateID = 8; even = false;}
    else if(iLink == 9)// a0b
      {crateID = 10; even = false;}
    else if(iLink == 10)// 80a
      {crateID = 8; even = true;}
    else if(iLink == 11)// b0b
      {crateID = 11; even = false;}
    else if(iLink == 12)// 30a
      {crateID = 3; even = true;}
    else if(iLink == 13)// 40a
      {crateID = 4; even = true;}
    else if(iLink == 14)// 30b
      {crateID = 3; even = false;}
    else if(iLink == 15)// a
      {crateID = 0; even = true;}
    else if(iLink == 16)// b
      {crateID = 0; even = false;}
    else if(iLink == 17)// 50a
      {crateID = 5; even = true;}
    else if(iLink == 18)// 10a
      {crateID = 1; even = true;}
    else if(iLink == 19)// 10b
      {crateID = 1; even = false;}
    else if(iLink == 20)// 20a
      {crateID = 2; even = true;}
    else if(iLink == 21)// 20b
      {crateID = 2; even = false;}
    else if(iLink == 22)// 50b
      {crateID = 5; even = false;}
    else if(iLink == 23)// 40b
      {crateID = 4; even = false;}
    else if(iLink == 24)// f0a
      {crateID = 15; even = true;}
    else if(iLink == 25)// 100a
      {crateID = 16; even = true;}
    else if(iLink == 26)// f0b
      {crateID = 15; even = false;}
    else if(iLink == 27)// c0a
      {crateID = 12; even = true;}
    else if(iLink == 28)// c0b
      {crateID = 12; even = false;}
    else if(iLink == 29)// 110a
      {crateID = 17; even = true;}
    else if(iLink == 30)// d0a
      {crateID = 13; even = true;}
    else if(iLink == 31)// d0b
      {crateID = 13; even = false;}
    else if(iLink == 32)// e0a
      {crateID = 14; even = true;}
    else if(iLink == 33)// e0b
      {crateID = 14; even = false;}
    else if(iLink == 34)// 110b
      {crateID = 17; even = false;}
    else if(iLink == 35)// 100b
      {crateID = 16; even = false;}
    return true;
  };

};

#endif
