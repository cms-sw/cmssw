#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "IORawData/RPCFileReader/interface/LinkDataXMLReader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include "boost/bind.hpp"
#include <boost/cstdint.hpp>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <fstream>

XERCES_CPP_NAMESPACE_USE

using namespace std;
using namespace edm;
using namespace rpcrawtodigi;

typedef uint64_t Word64;
/*############################################################################
#
#  Needed for xerces to work
#
############################################################################*/
string xMLCh2String(const XMLCh* ch) {
#ifdef __BORLANDC__
  if(ch == 0) return "";
  WideString wstr(ch);
  AnsiString astr(wstr);
  return astr.c_str();
#else
	if(ch == 0) return "";

	//auto_ptr<char> v(XMLString::transcode(ch));
  //return string(v.get());
  char* buf = XMLString::transcode(ch);
  string str(buf);
  XMLString::release(&buf);
  return str;
#endif
}
////////////////////////////////////////////////////////////
class XStr
{
public :
    XStr(const char* const toTranscode)
    {
        fUnicodeForm = XMLString::transcode(toTranscode);
    }

    ~XStr()
    {
        XMLString::release(&fUnicodeForm);
    }

    const XMLCh* unicodeForm() const
    {
        return fUnicodeForm;
    }

private :
    XMLCh*   fUnicodeForm;
};

#define X(str) XStr(str).unicodeForm()
#define Char2XMLCh(str) XStr(str).unicodeForm()
/*############################################################################*/
int LinkDataXMLReader::m_instanceCount = 0;
/*############################################################################
#
# Constructor
#
############################################################################*/
LinkDataXMLReader::LinkDataXMLReader(const edm::ParameterSet& iConfig,
				     InputSourceDescription const& desc):
  ExternalInputSource(iConfig, desc)
{


 //register products
  produces<FEDRawDataCollection>();
  //if do put with a label
  //produces<FEDRawDataCollection>("someLabel");

  eventCounter_=0; fileCounter_ = 0;
  run_ = 0; 
  event_ = 0;
  isOpen_ = false; noMoreData_ = false; 
  endOfFile_ = false;

  pair<int,int> rpcFEDS=FEDNumbering::getRPCFEDIds();
  for (int id= rpcFEDS.first; id<=rpcFEDS.second; ++id){
    results[id] = std::vector<rpcrawtodigi::EventRecords>();
  }

  if(m_instanceCount == 0) { 
    try {
      XMLPlatformUtils::Initialize();
      m_instanceCount++;
    }
    catch(const XMLException &toCatch)  {
      edm::LogError("LinkDataXMLReader")<< "Error during Xerces-c Initialization: " 
	+ xMLCh2String(toCatch.getMessage());
    }
  }  
}
/*############################################################################
#
#
############################################################################*/
LinkDataXMLReader::~LinkDataXMLReader(){}
/*############################################################################
#
#
############################################################################*/

// ------------ Method called to set run & event numbers  ------------
void  LinkDataXMLReader::setRunAndEventInfo(){

  bool triggeredOrEmpty = false;


  if(endOfFile_){
    isOpen_ = false;
    if(parser) {
      delete parser;
      parser = 0;
    }
  }

  if(!isOpen_){
    if(fileCounter_<(int)fileNames().size()){
      eventPos_[0]=0; eventPos_[1]=0;
      isOpen_=true;
      endOfFile_ = false;
      fileCounter_++;
      ++run_;
      event_ = 0;
      edm::LogInfo("LinkDataXMLReader")<< "[LinkDataXMLReader::setRunAndEventInfo] "
			   << "Open for reading file no. " << fileCounter_
			   << " " << fileNames()[fileCounter_-1].substr(5); 
      parser = XMLReaderFactory::createXMLReader();
      parser->setContentHandler(this);	
      parser->parseFirst(fileNames()[fileCounter_-1].substr(5).c_str(),aToken);
    }
    else{
      edm::LogInfo("LinkDataXMLReader")<< "[LinkDataXMLReader::setRunAndEventInfo] "
			   << "No more events to read. Finishing after " 
			   << eventCounter_ << " events.";
      noMoreData_=true;
      return;
    }
  }

  endOfEvent_ = false;
  while(parser->parseNext(aToken) &&  !endOfEvent_){}

    ++event_;
    ++eventCounter_;

  setRunNumber(run_);
  setEventNumber(event_);
  //setTime(mktime(&aTime));  
  return;
}
///////////////////////////////////////////////////////////////////////
bool LinkDataXMLReader::produce(edm::Event& ev){

  if(noMoreData_) return false;

  auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );

  pair<int,int> rpcFEDS=FEDNumbering::getRPCFEDIds();
  //pair<int,int> rpcFEDS(790,790);
  for (int id= rpcFEDS.first; id<=rpcFEDS.second; ++id){

    FEDRawData *  rawData1 =  rawData(id, results[id]);
    FEDRawData& fedRawData = buffers->FEDData(id);

    fedRawData = *rawData1;
  }
  ev.put( buffers );  

  
  return true;
}
/*############################################################################
#
#
############################################################################*/
void LinkDataXMLReader::startElement(const XMLCh* const uri,
				     const XMLCh* const localname,
				     const XMLCh* const qname,
				     const Attributes& attrs) {
  m_CurrElement = xMLCh2String(localname);
  //std::cout<<"Start element: "<<m_CurrElement<<endl;

 
  int rawData;
  uint16_t rawData1;

  if(m_CurrElement=="tc") triggerCrateNum = stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("num"))),1);
  if(m_CurrElement=="tb") triggerBoardNum = stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("num"))),1);
  if(m_CurrElement=="ol") opticalLinkNum = stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("num"))),1);  

  if(m_CurrElement=="lmd"){
    rawData = stringToInt(xMLCh2String(attrs.getValue(Char2XMLCh("raw"))),0);
 
    // BX 
    int trigger_BX =  200;
    int current_BX =  trigger_BX + 0;
    BXRecord bxr(current_BX);

    // TB 
    int tbLinkInputNumber = opticalLinkNum;    
    int rmb = getDCCInputChannelNum(triggerCrateNum, triggerBoardNum).second;
    cout<<"tc: "<<triggerCrateNum
	<<" tb: "<<triggerBoardNum
	<<" rmb: "<<rmb
        <<" fedID: "<<getDCCInputChannelNum(triggerCrateNum, triggerBoardNum).first
	<<" ol: "<< opticalLinkNum
	<<" raw data: "<<hex<<rawData<<dec<<endl;
    TBRecord tbr( tbLinkInputNumber, rmb);   

    // LB record
    typedef unsigned short Word16;
    RPCPacData linkData;
    linkData.fromRaw(rawData);
    rawData1 = (Word16(linkData.lbNum())<<14)
      |(Word16(linkData.partitionNum())<<10)
      |(Word16(linkData.endOfData())<<9)
      |(Word16(linkData.halfPartition())<<8)
      |(Word16(linkData.partitionData())<<0);

    LBRecord lbr(rawData1);
    cout<<"lbr.Data.lbNumber: "<<lbr.lbData().lbNumber()<<endl;
    cout<<"lbr.Data.lbData: "<<lbr.lbData().lbData()<<endl;
    cout<<"lbr.Data.partNumber: "<<lbr.lbData().partitionNumber()<<endl;
 
    int fedId = getDCCInputChannelNum(triggerCrateNum, triggerBoardNum).first;
    
    results[fedId].push_back(  EventRecords(trigger_BX, bxr, tbr, lbr) );
  }

}
/*############################################################################
#
#
############################################################################*/
void  LinkDataXMLReader::endElement(const XMLCh* const uri, const XMLCh* const localname, const XMLCh* const qname) {
 
  string m_CurrElement = xMLCh2String(localname);

  if(m_CurrElement=="bxData") endOfEvent_ = true; 
  if(m_CurrElement=="rpctDataStream") endOfFile_ = true; 
}



FEDRawData *  LinkDataXMLReader::rawData(int fedId,  const vector<EventRecords> & result){


  //
  // get merged records
  //
  int trigger_BX = 200;   // FIXME - set event by event but correct bx assigment in digi

  //
  // create data words
  //
  vector<Word64> dataWords;
  DataRecord empty;
  typedef vector<EventRecords>::const_iterator IR;
  for (IR ir = result.begin(), irEnd =  result.end() ; ir != irEnd; ++ir) {
    Word64 w = ( ( (Word64(ir->bxRecord().data()) << 16) | ir->tbRecord().data() ) << 16
                    | ir->lbRecord().data() ) << 16 | empty.data();
    dataWords.push_back(w);
  }

  //
  // create raw data
  //
  int nHeaders = 1;
  int nTrailers = 1;
  int dataSize = (nHeaders+nTrailers+dataWords.size()) * sizeof(Word64);
  FEDRawData * raw = new FEDRawData(dataSize);

  //
  // add header
  //
  unsigned char *pHeader  = raw->data();
  int evt_ty = 3;
  int lvl1_ID = 100; // FIXME - increase/set event by event
  int source_ID = fedId;
  FEDHeader::set(pHeader, evt_ty, lvl1_ID, trigger_BX, source_ID);

  //
  // add datawords
  //
  for (unsigned int idata = 0; idata < dataWords.size(); idata ++) {
    Word64 * word = reinterpret_cast<Word64* >(pHeader+(idata+1)*sizeof(Word64));
    *word = dataWords[idata];
  }

  //
  // add trailer
  //
  unsigned char *pTrailer = pHeader + raw->size()-sizeof(Word64);
  int crc = 0;
  int evt_stat = 15;
  int tts = 0;
  int datasize =  raw->size()/sizeof(Word64);
  FEDTrailer::set(pTrailer, datasize, crc, evt_stat, tts);

  return raw;
}
/*############################################################################
#
#
############################################################################*/


std::string LinkDataXMLReader::IntToString(int i, int opt){

 std::stringstream ss;
 if(opt==1) ss << std::hex << i << std::dec;
 else ss << i ;

 return ss.str();

}

int  LinkDataXMLReader::stringToInt(std::string str, int opt) {

  int number;
  std::stringstream lineStream(str);
  if(opt==0) lineStream>>hex>>number;
  if(opt==1) lineStream>>dec>>number;
  return number;

  for(unsigned int i = 0; i < str.size(); i++)
    if(str[i] < '0' || str[i] > '9')
       edm::LogError("LinkDataXMLReader")<< "Error in stringToInt(): the string: "
					 <<str<<" cannot be converted to a number";
  return atoi(str.c_str());
}


void  LinkDataXMLReader::clear(){ 

 pair<int,int> rpcFEDS=FEDNumbering::getRPCFEDIds();
  for (int id= rpcFEDS.first; id<=rpcFEDS.second; ++id){
    results[id].clear();
  }
}


std::pair<int,int>  LinkDataXMLReader::getDCCInputChannelNum(int tcNumber, int tbNumber){

  int fedNumber = 790 + tcNumber/4;
  int dccInputChannelNum = tbNumber + 9*(tcNumber%4);

  return std::pair<int,int>(fedNumber,dccInputChannelNum);
}
