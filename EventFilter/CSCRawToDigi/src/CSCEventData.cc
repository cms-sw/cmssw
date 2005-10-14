#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#ifndef UNPCK_ONLY
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#endif
#include "EventFilter/CSCRawToDigi/interface/CSCDMBTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBData.h"

//remove this COBRA vestige
//#include "Utilities/GenUtil/interface/BitVector.h"

#include <iostream>
#include <iterator>


bool CSCEventData::debug = false;

CSCEventData::CSCEventData(int chamberType) : 
  theDMBHeader(), 
  theALCTHeader(0), 
  theAnodeData(0),
  theALCTTrailer(0),
  theTMBData(0),
  theDMBTrailer(),
  theChamberType(chamberType)
{
  for(unsigned i = 0; i < 5; ++i) {
    theCFEBData[i] = 0;
  }
}


CSCEventData::CSCEventData(unsigned short * buf)
 {
    // zero everything
    init();
    unsigned short * pos = buf;
    if(debug) {
      std::cout << "The event data " << std::endl;
      for(int i = 0; i < 16; ++i) {
	std::cout << std::hex << pos[i ] << " " ;
        if((i+1)%4 == 0) std::cout << std::endl;
      }
    }
   
    theDMBHeader = CSCDMBHeader(pos);
    if(!(theDMBHeader.check())) {
      std::cout << "Bad DMB Header??? " << std::endl;
      std::cout << "first four words " << std::endl;
      for(int i = 0; i < 4; ++i) {
	 std::cout << std::hex << pos[i ] << " " ;
      }
      std::cout << std::dec << std::endl;
    }
  
    nalct_ = theDMBHeader.nalct();
    nclct_ = theDMBHeader.nclct();
   
      
    if (debug){
      //std::cout << theDMBHeader;
      std::cout << "nalct = " << nalct() << std::endl;
      std::cout << "nclct = " << nclct() << std::endl;
    }

    if (debug) {
      std::cout << "size in words of DMBHeader" << theDMBHeader.sizeInWords() << std::endl;
      std::cout << "sizeof(DMBHeader)" << sizeof(theDMBHeader) << std::endl;      
    }
   
    pos += theDMBHeader.sizeInWords();

    if (nalct() > 0) {
      theALCTHeader = new CSCALCTHeader( pos );
      //std::cout << *theALCTHeader << std::endl;
      //std::cout << "first ALCT header word " << std::hex << *pos << std::endl;
  
      pos += theALCTHeader->sizeInWords(); //size of the header
      if (debug) std::cout << "size of ALCTHeader = "<< theALCTHeader->sizeInWords() << std::endl;     
      if(!theALCTHeader->check()) {  
         std::cout <<"+++WARNING: Corrupt ALCT data - won't attempt to decode" << std::endl;
 
      } else {
        theAnodeData = new CSCAnodeData(*theALCTHeader, pos);  
        pos += theAnodeData->sizeInWords(); // size of the data is determined during unpacking
        theALCTTrailer = new CSCALCTTrailer( pos );
        pos += theALCTTrailer->sizeInWords();
        if (debug) std::cout << "size of AnodeData = " << theAnodeData->sizeInWords() << std::endl;
        if (debug) std::cout << "size of ALCTTrailer = " << theALCTTrailer->sizeInWords() << std::endl;
      }
    }

    //std::cout << "len after ALCT " << pos-buf << std::endl ;
    if (nclct() > 0) {
      theTMBData = new CSCTMBData(pos);  //fill all TMB data
      pos += theTMBData->size();
      if (debug) std::cout << "size of TMBData = " << theTMBData->size() << std::endl;
    }
    //std::cout << "len after TMB " << pos-buf << std::endl ;
    //
    for(int icfeb = 0; icfeb < 5; ++icfeb) {
      theCFEBData[icfeb] = 0;
      int cfeb_available = theDMBHeader.cfebAvailable(icfeb);
      // There seemed to be a bug in the DMB firmware used during 2003 test
      // beam, which resulted in occasional zero values of cfebAvailable
      // word in DMBHeader whereas in reality it was not zero.  So TMBHeader
      // might be a better place to get this info from.  But I am not quite
      // sure of it, so I leave it as is for now. -Sl.
      // int cfeb_available = 0;
      // if (nclct() > 0)
      //   cfeb_available = (theTMBData->tmbHeader().ActiveCFEBs() >> icfeb) & 1;
      if (cfeb_available) {
	// Fill CFEB data and convert it into cathode digis
	if(debug) {
	  std::cout << "unpacking CFEB " << icfeb << std::endl;
	  for(unsigned i = 0; i < 4; ++i) {
	    std::cout << i << std::hex << " " << pos[i] << std::dec << std::endl;
	  }
	  for(unsigned i = 96; i < 100; ++i) {
	    std::cout << i << std::hex << " " << pos[i] << std::dec << std::endl;
	  }
	}
	theCFEBData[icfeb] = new CSCCFEBData(icfeb, pos);
	theCFEBData[icfeb]->check();
	pos += theCFEBData[icfeb]->sizeInWords();
	if (debug) {
	  std::cout << "size in words of CFEB =" << theCFEBData[icfeb]->sizeInWords() << std::endl;
	  std::cout << "sizeof (CFEB) = " << sizeof(* theCFEBData[icfeb]) << std::endl;
	}
      }
    }
    
    if(debug) {
      std::cout << "ready to unpack trailer " << std::endl;
      for(unsigned i = 0; i < 8; ++i) {
	std::cout << std::hex << pos[i] <<" ";
	if((i+1)%4 == 0) std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    theDMBTrailer = *( (CSCDMBTrailer *) pos );
    pos += theDMBTrailer.sizeInWords();
    size_ = pos-buf;
 }


CSCEventData::CSCEventData(const CSCEventData & data) {
  copy(data);
}

CSCEventData::~CSCEventData() {
  destroy();
}


CSCEventData CSCEventData::operator=(const CSCEventData & data) {
  // check for self-assignment before destructing
  if(&data != this) destroy();
  copy(data);
  return *this;
}


void CSCEventData::init() {
  theALCTHeader = 0;
  theAnodeData = 0;
  theALCTTrailer = 0;
  theTMBData = 0;
  for(int icfeb = 0; icfeb < 5; ++icfeb) {
    theCFEBData[icfeb] = 0;
  }
}


void CSCEventData::copy(const CSCEventData & data) {
  init();
  theDMBHeader  = data.theDMBHeader;
  theDMBTrailer = data.theDMBTrailer;
  if(data.theALCTHeader != NULL)
    theALCTHeader  = new CSCALCTHeader(*(data.theALCTHeader));
  if(data.theAnodeData != NULL) 
    theAnodeData   = new CSCAnodeData(*(data.theAnodeData));
  if(data.theALCTTrailer != NULL) 
    theALCTTrailer = new CSCALCTTrailer(*(data.theALCTTrailer));
  if(data.theTMBData != NULL) 
    theTMBData     = new CSCTMBData(*(data.theTMBData));
  for(int icfeb = 0; icfeb < 5; ++icfeb) {
    theCFEBData[icfeb] = 0;
    if(data.theCFEBData[icfeb] != NULL) 
      theCFEBData[icfeb] = new CSCCFEBData(*(data.theCFEBData[icfeb]));
  }   
  nalct_ = data.nalct_;
  nclct_ = data.nclct_;
  size_  = data.size_;
  theChamberType = data.theChamberType;
}


void CSCEventData::destroy() {
  delete theALCTHeader;
  delete theAnodeData;
  delete theALCTTrailer;
  delete theTMBData;
  for(int icfeb = 0; icfeb < 5; ++icfeb) {
    delete theCFEBData[icfeb];
  }
}

#ifndef UNPCK_ONLY

std::vector<CSCStripDigi> CSCEventData::stripDigis(unsigned ilayer) const {
  assert(ilayer > 0 && ilayer <= 6);
  std::vector<CSCStripDigi> result;
  for(unsigned icfeb = 0; icfeb < 5; ++icfeb) {
    if(theCFEBData[icfeb] != NULL) {
      std::vector<CSCStripDigi> newDigis = theCFEBData[icfeb]->digis(ilayer);
      result.insert(result.end(), newDigis.begin(), newDigis.end());
    }
  }

  // now add the comparator information
  addComparatorInformation(result, ilayer);

  if (debug) {
    for (std::vector<CSCStripDigi>::iterator pdigi = result.begin();
	 pdigi != result.end(); pdigi++) {
      if (pdigi->getComparatorOutput() > 0)
	std::std::cout << "Layer " << ilayer << " " << *pdigi;
    }
  }

  return result;
}

std::vector<CSCWireDigi> CSCEventData::wireDigis(unsigned ilayer) const {
  if(theAnodeData == 0) {
    return std::vector<CSCWireDigi>();
  } else {
    return theAnodeData->wireDigis(ilayer);
  }
}


std::vector < std::vector<CSCStripDigi> > CSCEventData::stripDigis() const {
  std::vector < std::vector<CSCStripDigi> > result;
  for (int layer = 1; layer <= 6; ++layer) {
    std::vector<CSCStripDigi> digis = stripDigis(layer);

    // We already added comparator information in stripDigis(layer) routine.
    // Please check Rick. -JM
    //!!!addComparatorInformation(digis, layer);
    result.push_back(digis);
  }
  return result;
}

std::vector < std::vector<CSCWireDigi> > CSCEventData::wireDigis() const {
  std::vector < std::vector<CSCWireDigi> > result;
  for (int layer = 1; layer <= 6; ++layer) {
    result.push_back(wireDigis(layer));
  }
  return result;
}

#endif

CSCCFEBData* CSCEventData::cfebData(unsigned icfeb) const {
  return theCFEBData[icfeb];
}


CSCALCTHeader CSCEventData::alctHeader() const{
   if(nalct() == 0) throw("No ALCT for this chamber");
   return *theALCTHeader;
}

CSCALCTTrailer CSCEventData::alctTrailer() const{
   if(nalct() == 0) throw("No ALCT for this chamber");
   return *theALCTTrailer;
}


CSCAnodeData & CSCEventData::alctData() const {
   if(nalct() == 0) throw("No ALCT for this chamber");
   return *theAnodeData;
}

CSCTMBData & CSCEventData::tmbData() const {
  if(nclct() == 0) throw("No CLCT for this chamber");
   return *theTMBData;
}


CSCTMBHeader & CSCEventData::tmbHeader() const {
  if(nclct() == 0) throw("No CLCT for this chamber");
  return tmbData().tmbHeader();
}

CSCCLCTData & CSCEventData::clctData() const {
  if(nclct() == 0) throw("No CLCT for this chamber");
  return tmbData().clctData();
}


#ifndef UNPCK_ONLY
void CSCEventData::add(const CSCStripDigi & digi, int layer) {
  //@@ need special logic here for ME11
  unsigned cfeb = (digi.channel()-1)/16;
  //@@ obviously a hack here
  bool sixteenSamples = false;
  if(theCFEBData[cfeb] == 0) {
    theCFEBData[cfeb] = new CSCCFEBData(cfeb, sixteenSamples);
    theDMBHeader.addCFEB(cfeb);
  }
  theCFEBData[cfeb]->add(digi, layer);

  // if there's a comparator output, put it in the TMB data
  int comparatorOutput = digi.getComparatorOutput();
  if(comparatorOutput > 0) {
    float t = digi.getComparatorTime();
    add(CSCComparatorOutput(t, comparatorOutput, digi.channel()/2, t+250.), layer);
  }
}


void CSCEventData::add(const CSCWireDigi & digi, int layer) {
  if(theAnodeData == NULL) {
    createALCTClasses();
  }
  theAnodeData->add(digi, layer);
}


void CSCEventData::add(const CSCComparatorOutput & output, int layer) {
  if(theTMBData == NULL) {
    theTMBData = new CSCTMBData();
    theTMBData->tmbHeader().setEventInformation(theDMBHeader);
    // set data available flag
    theDMBHeader.addNCLCT();

  }
  theTMBData->clctData().add(output, layer);
}

#endif

void CSCEventData::setEventInformation(int bxnum, int lvl1num) {
  theDMBHeader.setBXN(bxnum);
  theDMBHeader.setL1A(lvl1num);
  if(theALCTHeader) {
    theALCTHeader->setEventInformation(theDMBHeader);
  }
  if(theTMBData) {
    theTMBData->tmbHeader().setEventInformation(theDMBHeader);
  }
}
    

void CSCEventData::createALCTClasses() {
  assert(theChamberType>0);
  theALCTHeader = new CSCALCTHeader(theChamberType);
  theALCTHeader->setEventInformation(theDMBHeader);
  theAnodeData = new CSCAnodeData(*theALCTHeader);
  theALCTTrailer = new CSCALCTTrailer();
  // set data available flag
  theDMBHeader.addNALCT();

}

#ifndef UNPCK_ONLY

void CSCEventData::addComparatorInformation(std::vector<CSCStripDigi>& digis, int ilayer) const {
  if(theTMBData != 0) {
    std::vector<CSCComparatorOutput> comparatorOutputs = theTMBData->clctData().comparatorOutputs(ilayer);
    int ncmp = comparatorOutputs.size();
    for(unsigned int i = 0; i < digis.size(); ++i) {
      int strip = digis[i].channel();
      // only fill in results for even strips
      if(strip%2 == 0) {
        // try to find a comparator for this digi
        bool found = false;
        for(int icmp = 0; !found && icmp < ncmp; ++icmp) {
          // elements count from zero
          if(comparatorOutputs[icmp].element() == (strip-1)/2) {
             digis[i].setComparatorOutput(comparatorOutputs[icmp].output());
             digis[i].setComparatorTime((int)comparatorOutputs[icmp].time());

	     // Is it okay to add this to the digis?  -JM
             digis[i].setBeamCrossingTag((int)comparatorOutputs[icmp].time());
             found = true;
          }
        }
      }
    } // loop over digis
  }
}

#endif

/*
pair<int, unsigned short *> CSCEventData::pack() {
  unsigned short * data = new unsigned short[20000];

  memcpy(data, &theDMBHeader, theDMBHeader.sizeInWords()*2);
  int pos = theDMBHeader.sizeInWords();
  std::cout << "added theDMBHeaqder " << pos << std::endl;

  if(theALCTHeader != NULL) {
    memcpy(data+pos, theALCTHeader->data(), theALCTHeader->sizeInWords()*2);
    pos += theALCTHeader->sizeInWords();
    std::cout << "adding ALCT Header " << std::endl;
  }
  if(theAnodeData != NULL) {
    memcpy(data+pos, theAnodeData->data(), theAnodeData->sizeInWords()*2);
    pos += theAnodeData->sizeInWords();
  }
  if(theALCTTrailer != NULL) {
    memcpy(data+pos, theALCTTrailer->data(), theALCTTrailer->sizeInWords()*2);
     std::cout << "putinng ALCTTRAILER at " << pos << std::endl;
    pos += theALCTTrailer->sizeInWords();
  }
  std::cout << "added ALCT " << pos << std::endl;

  if(theTMBData != NULL) {
    BitVector tmbPacked = theTMBData->pack();
    memcpy(data+pos, tmbPacked.rawData(), tmbPacked.nBits()/8);
    pos += tmbPacked.nBits()/16;
  }

  std::cout << "added TMB " << pos << std::endl;
  for(int icfeb = 0;  icfeb < 5;  ++icfeb) {
    if(theCFEBData[icfeb] != NULL) {
      memcpy(data+pos, theCFEBData[icfeb]->data(), theCFEBData[icfeb]->sizeInWords()*2);
      pos += theCFEBData[icfeb]->sizeInWords();
    }
  }

  memcpy(data+pos, &theDMBTrailer, theDMBTrailer.sizeInWords()*2);
  pos += theDMBTrailer.sizeInWords();

  return pair<int, unsigned short *>(pos, data);
}


void printVector(const BitVector & v) {
  unsigned short * buf = (unsigned short *) v.rawData();
  int sizeInWords = v.size()/16;
  int sizeInLines = sizeInWords / 4;
  for (int i=0;i<sizeInLines;i++) {
    printf("%d   %04x %04x %04x %04x\n",i,buf[4*i+3],buf[4*i+2],buf[4*i+1],buf[4*i]);
  }
}


BitVector CSCEventData::packVector() {
  BitVector result((const unsigned int*) &theDMBHeader, 
                   theDMBHeader.sizeInWords()*16);
  if(theALCTHeader != NULL) {
    BitVector alctHeader((const unsigned int*) theALCTHeader->data(),
                         theALCTHeader->sizeInWords()*16);
    result.assign(result.nBits(), alctHeader.nBits(), alctHeader);
    if(debug) std::cout << "adding ALCT Header " << std::endl;
  }

  if(theAnodeData != NULL) {
    BitVector anodeData((const unsigned int*) theAnodeData->data(),
                         theAnodeData->sizeInWords()*16);
    result.assign(result.nBits(), anodeData.nBits(), anodeData);
  }
  if(theALCTTrailer != NULL) {
    BitVector alctTrailer((const unsigned int*) &theALCTTrailer,
                         theALCTTrailer->sizeInWords()*16);
    result.assign(result.nBits(), alctTrailer.nBits(), alctTrailer);
    if(debug) std::cout << "adding ALCT Trailer " << std::endl;
  }
  if(theTMBData != NULL) {
    BitVector tmbData(theTMBData->pack());
    result.assign(result.nBits(), tmbData.nBits(), tmbData);

  }

  if(debug) std::cout << "adding CFEB " << std::endl;
  for(int icfeb = 0;  icfeb < 5;  ++icfeb) {
    if(theCFEBData[icfeb] != NULL) {
      BitVector cfeb((const unsigned int*) theCFEBData[icfeb]->data(), 
                      theCFEBData[icfeb]->sizeInWords()*16);
      result.assign(result.nBits(), cfeb.nBits(), cfeb);
    }
  }
  if(debug) std::cout << "adding DMBTrailer " << std::endl;

  BitVector dmbTrailer((const unsigned int*) &theDMBTrailer,
                 theDMBTrailer.sizeInWords()*16);
  result.assign(result.nBits(), dmbTrailer.nBits(), dmbTrailer);
  if(debug) printVector(result);

  return result;
}

*/

#ifndef UNPCK_ONLY

std::ostream & operator<<(std::ostream & os, const CSCEventData & evt) {
  for(int ilayer = 1; ilayer <= 6; ++ilayer) {
    std::vector<CSCStripDigi> stripDigis = evt.stripDigis(ilayer);
    copy(stripDigis.begin(), stripDigis.end(), std::ostream_iterator<CSCStripDigi>(os, "\n"));

    std::vector<CSCWireDigi> wireDigis = evt.wireDigis(ilayer);
    copy(wireDigis.begin(), wireDigis.end(), std::ostream_iterator<CSCWireDigi>(os, "\n"));
  }
  return os;
}

#endif
