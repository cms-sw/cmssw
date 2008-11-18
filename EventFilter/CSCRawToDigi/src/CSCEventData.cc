#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/CSCRawToDigi/src/cscPackerCompare.h"
#include <iostream>
#include <iterator>
#include "EventFilter/CSCRawToDigi/src/bitset_append.h"
#include "FWCore/Utilities/interface/Exception.h"


bool CSCEventData::debug = false;

CSCEventData::CSCEventData(int chamberType) : 
  theDMBHeader(), 
  theALCTHeader(0), 
  theAnodeData(0),
  theALCTTrailer(0),
  theTMBData(0),
  theDMBTrailer(),
  theChamberType(chamberType){
  
  for(unsigned i = 0; i < 5; ++i) {
    theCFEBData[i] = 0;
  }
}


CSCEventData::CSCEventData(unsigned short * buf){
  // zero everything
  init();
  unsigned short * pos = buf;
  if(debug)    {
    LogTrace ("CSCEventData|CSCRawToDigi") << "The event data ";
    for(int i = 0; i < 16; ++i){
      LogTrace ("CSCEventData|CSCRawToDigi") << std::hex << pos[i ] << " ";
    }
    }
   
  theDMBHeader = CSCDMBHeader(pos);
  if(!(theDMBHeader.check())) {
    LogTrace ("CSCEventData|CSCRawToDigi")  << "Bad DMB Header??? " << " first four words: ";
    for(int i = 0; i < 4; ++i){
      LogTrace ("CSCEventData|CSCRawToDigi") << std::hex << pos[i ] << " ";
    }
  }
  
      
  if (debug) {
    LogTrace ("CSCEventData|CSCRawToDigi") << "nalct = " << nalct();
    LogTrace ("CSCEventData|CSCRawToDigi") << "nclct = " << nclct();
  }

  if (debug)  {
    LogTrace ("CSCEventData|CSCRawToDigi") << "size in words of DMBHeader" << theDMBHeader.sizeInWords();
    LogTrace ("CSCEventData|CSCRawToDigi") << "sizeof(DMBHeader)" << sizeof(theDMBHeader); 
  }
   
  pos += theDMBHeader.sizeInWords();

  if (nalct() ==1)  {
    if (isALCT(pos)) {//checking for ALCTData
      theALCTHeader = new CSCALCTHeader( pos );
      if(!theALCTHeader->check()){  
	LogTrace ("CSCEventData|CSCRawToDigi") <<"+++WARNING: Corrupt ALCT data - won't attempt to decode";
      } 
      else {
	//dataPresent|=0x40;
	pos += theALCTHeader->sizeInWords(); //size of the header
	//fill ALCT Digis
	theALCTHeader->ALCTDigis();    
	theAnodeData = new CSCAnodeData(*theALCTHeader, pos);  
	pos += theAnodeData->sizeInWords(); // size of the data is determined during unpacking
	theALCTTrailer = new CSCALCTTrailer( pos );
	pos += theALCTTrailer->sizeInWords();
      }
    } 
    else {
      LogTrace ("CSCEventData|CSCRawToDigi") << "Error:nalct reported but no ALCT data found!!!";
    }
  }

  if (nclct() ==1)  {
    if (isTMB(pos)) {
      //dataPresent|=0x20;
      theTMBData = new CSCTMBData(pos);  //fill all TMB data
      pos += theTMBData->size();
    }
    else {
      LogTrace ("CSCEventData|CSCRawToDigi") << "Error:nclct reported but no TMB data found!!!";
    }
  }

  //now let's try to find and unpack the DMBTrailer
  bool dmbTrailerReached= false;
  for (int i=0; i<12000; ++i) {//8000 max for cfeb + 1980ALCT + 287 TMB
    dmbTrailerReached =
      (*(i+pos) & 0xF000) == 0xF000 && (*(i+pos+1) & 0xF000) == 0xF000
      && (*(i+pos+2) & 0xF000) == 0xF000 && (*(i+pos+3) & 0xF000) == 0xF000
      && (*(i+pos+4) & 0xF000) == 0xE000 && (*(i+pos+5) & 0xF000) == 0xE000
      && (*(i+pos+6) & 0xF000) == 0xE000 && (*(i+pos+7) & 0xF000) == 0xE000;
    if (dmbTrailerReached) {
      theDMBTrailer = *( (CSCDMBTrailer *) (pos+i) );
      break;
    }
  }
  if (dmbTrailerReached) {
    for(int icfeb = 0; icfeb < 5; ++icfeb)  {
      theCFEBData[icfeb] = 0;
      int cfeb_available = theDMBHeader.cfebAvailable(icfeb);
      unsigned int cfebTimeout = theDMBTrailer.cfeb_starttimeout | theDMBTrailer.cfeb_endtimeout;    
      //cfeb_available cannot be trusted - need additional verification!
      if ( cfeb_available==1 )   {
	if ((cfebTimeout >> icfeb) & 1) {
	  if (debug) LogTrace ("CSCEventData|CSCRawToDigi") << "CFEB Timed out! ";
	} else {
	  //dataPresent|=(0x1>>icfeb);
	  // Fill CFEB data and convert it into cathode digis
	  theCFEBData[icfeb] = new CSCCFEBData(icfeb, pos);
	  pos += theCFEBData[icfeb]->sizeInWords();
	}
      }
    }   
    pos += theDMBTrailer.sizeInWords();
    size_ = pos-buf;
  }
  else {
    LogTrace ("CSCEventData|CSCRawToDigi") << "Critical Error: DMB Trailer was not found!!! ";
  }
}

bool CSCEventData::isALCT(const short unsigned int * buf) {
  return (((buf[0]&0xFFFF)==0xDB0A)||(((buf[0]&0xF800)==0x6000)&&((buf[1]&0xF800)==0)));
}

bool CSCEventData::isTMB(const short unsigned int * buf) {
  return ((buf[0]&0xFFF)==0xB0C);
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
  //dataPresent = 0;
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


std::vector<CSCStripDigi> CSCEventData::stripDigis(unsigned ilayer) const {
  assert(ilayer > 0 && ilayer <= 6);
  std::vector<CSCStripDigi> result;
  for(unsigned icfeb = 0; icfeb < 5; ++icfeb){
    if(theCFEBData[icfeb] != NULL) {
      std::vector<CSCStripDigi> newDigis = theCFEBData[icfeb]->digis(ilayer);
      result.insert(result.end(), newDigis.begin(), newDigis.end());
    }
  }
  
  return result;
}


std::vector<CSCStripDigi> CSCEventData::stripDigis(unsigned idlayer, unsigned icfeb) const {
  //  assert(ilayer > 0 && ilayer <= 6); // off because now idlayer is raw cscdetid
  std::vector<CSCStripDigi> result;
  if(theCFEBData[icfeb] != NULL) {
    std::vector<CSCStripDigi> newDigis = theCFEBData[icfeb]->digis(idlayer);
    result.insert(result.end(), newDigis.begin(), newDigis.end());
  }
  

  return result;
}


std::vector<CSCWireDigi> CSCEventData::wireDigis(unsigned ilayer) const {
  if(theAnodeData == 0)    {
    return std::vector<CSCWireDigi>();
  } 
  else    {
    return theAnodeData->wireDigis(ilayer);
  }
}


std::vector < std::vector<CSCStripDigi> > CSCEventData::stripDigis() const {
  std::vector < std::vector<CSCStripDigi> > result;
  for (int layer = 1; layer <= 6; ++layer) {
    std::vector<CSCStripDigi> digis = stripDigis(layer);
    result.push_back(digis);
  }
  return result;
}

std::vector < std::vector<CSCWireDigi> > CSCEventData::wireDigis() const {
  std::vector < std::vector<CSCWireDigi> > result;
  for (int layer = 1; layer <= 6; ++layer)     {
    result.push_back(wireDigis(layer));
  }
  return result;
}


CSCCFEBData* CSCEventData::cfebData(unsigned icfeb) const {
  return theCFEBData[icfeb];
}


CSCALCTHeader* CSCEventData::alctHeader() const{
  if(nalct() == 0) throw cms::Exception("No ALCT for this chamber");
  return theALCTHeader;
}

CSCALCTTrailer * CSCEventData::alctTrailer() const{
  if(nalct() == 0) throw cms::Exception("No ALCT for this chamber");
  return theALCTTrailer;
}


CSCAnodeData * CSCEventData::alctData() const {
  if(nalct() == 0) throw cms::Exception("No ALCT for this chamber");
  return theAnodeData;
}

CSCTMBData * CSCEventData::tmbData() const {
  if(nclct() == 0) throw cms::Exception("No CLCT for this chamber");
  return theTMBData;
}


CSCTMBHeader * CSCEventData::tmbHeader() const {
  if((nclct() == 0)||(tmbData()==NULL)) throw cms::Exception("No CLCT header for this chamber");
  return tmbData()->tmbHeader();
}

CSCCLCTData * CSCEventData::clctData() const {
  if((nclct() == 0)||(tmbData()==NULL)) throw cms::Exception("No CLCT data for this chamber");
  return tmbData()->clctData();
}


void CSCEventData::setEventInformation(int bxnum, int lvl1num) {
  theDMBHeader.setBXN(bxnum);
  theDMBHeader.setL1A(lvl1num);
  if(theALCTHeader)     {
    theALCTHeader->setEventInformation(theDMBHeader);
  }
  if(theTMBData)  {
    theTMBData->tmbHeader()->setEventInformation(theDMBHeader);
  }
}
    

void CSCEventData::checkALCTClasses() {
  if(theAnodeData == NULL)
  {
    assert(theChamberType>0);
    theALCTHeader = new CSCALCTHeader(theChamberType);
    theALCTHeader->setEventInformation(theDMBHeader);
    theAnodeData = new CSCAnodeData(*theALCTHeader);
    theALCTTrailer = new CSCALCTTrailer();
    // set data available flag
    theDMBHeader.addNALCT();
  }
}


void CSCEventData::checkTMBClasses() 
{
  if(theTMBData == NULL)    {
    theTMBData = new CSCTMBData();
    theTMBData->tmbHeader()->setEventInformation(theDMBHeader);
    theDMBHeader.addNCLCT();
  }
}


void CSCEventData::add(const CSCStripDigi & digi, int layer) {
  //@@ need special logic here for ME11
  unsigned cfeb = (digi.getStrip()-1)/16;
  bool sixteenSamples = false;
  if (digi.getADCCounts().size()==16) sixteenSamples = true;  
  if(theCFEBData[cfeb] == 0)    {
    theCFEBData[cfeb] = new CSCCFEBData(cfeb, sixteenSamples);
    theDMBHeader.addCFEB(cfeb);
  }
  theCFEBData[cfeb]->add(digi, layer);
}


void CSCEventData::add(const CSCWireDigi & digi, int layer) {
  checkALCTClasses();
  theAnodeData->add(digi, layer);
}

void CSCEventData::add(const CSCComparatorDigi & digi, int layer) {
  checkTMBClasses();
  theTMBData->clctData()->add(digi, layer);
}



void CSCEventData::add(const std::vector<CSCALCTDigi> & digis) {
  checkALCTClasses();
  theALCTHeader->add(digis);
}


void CSCEventData::add(const std::vector<CSCCLCTDigi> & digis) {
  checkTMBClasses();
  theTMBData->tmbHeader()->add(digis);
}

void CSCEventData::add(const std::vector<CSCCorrelatedLCTDigi> & digis) {
  checkTMBClasses();
  theTMBData->tmbHeader()->add(digis);
}




std::ostream & operator<<(std::ostream & os, const CSCEventData & evt) {
  for(int ilayer = 1; ilayer <= 6; ++ilayer)     {
    std::vector<CSCStripDigi> stripDigis = evt.stripDigis(ilayer);
    //copy(stripDigis.begin(), stripDigis.end(), std::ostream_iterator<CSCStripDigi>(os, "\n"));
    //print your scas here
    std::vector<CSCWireDigi> wireDigis = evt.wireDigis(ilayer);
    //copy(wireDigis.begin(), wireDigis.end(), std::ostream_iterator<CSCWireDigi>(os, "\n"));
  }
  return os;
}

boost::dynamic_bitset<> CSCEventData::pack() {
  boost::dynamic_bitset<> result = bitset_utilities::ushortToBitset( theDMBHeader.sizeInWords()*16, 
								     theDMBHeader.data());

  if(theALCTHeader != NULL)     {
    boost::dynamic_bitset<> alctHeader = bitset_utilities::ushortToBitset(theALCTHeader->sizeInWords()*16,
									  theALCTHeader->data());
    result = bitset_utilities::append(result, alctHeader);
  }
  if(theAnodeData != NULL) {
    boost::dynamic_bitset<> anodeData = bitset_utilities::ushortToBitset (theAnodeData->sizeInWords()*16,
									  theAnodeData->data());
    result = bitset_utilities::append(result, anodeData);
  }
  if(theALCTTrailer != NULL)  {
    boost::dynamic_bitset<> alctTrailer =bitset_utilities::ushortToBitset(theALCTTrailer->sizeInWords()*16,
									  theALCTTrailer->data());
    result = bitset_utilities::append(result, alctTrailer);
  }

  if(theTMBData != NULL)  {
    result  = bitset_utilities::append(result, theTMBData->pack());
  }

  for(int icfeb = 0;  icfeb < 5;  ++icfeb)  {
    if(theCFEBData[icfeb] != NULL){
      boost::dynamic_bitset<> cfebData = bitset_utilities::ushortToBitset(theCFEBData[icfeb]->sizeInWords()*16,
									  theCFEBData[icfeb]->data());
      result = bitset_utilities::append(result, cfebData);
    }
  }
  
  boost::dynamic_bitset<> dmbTrailer = bitset_utilities::ushortToBitset( theDMBTrailer.sizeInWords()*16,
									 theDMBTrailer.data());
  result = bitset_utilities::append(result, dmbTrailer);
  return result;
}


void CSCEventData::selfTest() {
  CSCEventData chamberData(5);
  CSCDetId detId(1, 3, 2, 1, 3);
  std::vector<CSCCLCTDigi> clctDigis;
  clctDigis.push_back(CSCCLCTDigi(1, 1, 4, 1, 0, 30, 3, 0, 1)); // valid for 2007
  clctDigis.push_back(CSCCLCTDigi(1, 1, 2, 1, 1, 31, 1, 2, 2));
  
  // BX of LCT (8th argument) is 1-bit word (the least-significant bit
  // of ALCT's bx).
  std::vector<CSCCorrelatedLCTDigi> corrDigis;
  corrDigis.push_back(CSCCorrelatedLCTDigi(1, 1, 2, 10, 98, 5, 0, 1, 0, 0, 0, 0));
  corrDigis.push_back(CSCCorrelatedLCTDigi(2, 1, 2, 20, 15, 9, 1, 0, 0, 0, 0, 0));

  chamberData.add(clctDigis);
  chamberData.add(corrDigis);

  CSCWireDigi wireDigi(10, 6);
  CSCComparatorDigi comparatorDigi(30, 1, 6);
  chamberData.add(wireDigi, 3);
  chamberData.add(comparatorDigi, 3);

  CSCEventData newData = cscPackAndUnpack(chamberData);

  std::vector<CSCCLCTDigi> clcts = newData.tmbHeader()->CLCTDigis(detId.rawId());
  assert(cscPackerCompare(clcts[0],clctDigis[0]));
  assert(cscPackerCompare(clcts[1],clctDigis[1]));

  std::vector<CSCCorrelatedLCTDigi> lcts = newData.tmbHeader()->CorrelatedLCTDigis(detId.rawId());
  assert(cscPackerCompare(lcts[0], corrDigis[0]));
  assert(cscPackerCompare(lcts[1], corrDigis[1]));

}
 
