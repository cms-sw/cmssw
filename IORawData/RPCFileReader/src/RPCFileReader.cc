
#include <stdexcept>
#include <memory>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <cctype>
#include <vector>
#include <bitset>

#include "IORawData/RPCFileReader/interface/RPCFileReader.h"
#include "IORawData/RPCFileReader/interface/RPCPacData.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "interface/shared/include/fed_header.h"
#include "interface/shared/include/fed_trailer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace edm;

//Ctor
RPCFileReader::RPCFileReader(ParameterSet const& pset, 
			     InputSourceDescription const& desc):
  ExternalInputSource(pset, desc),
  theLogCones_(12),linkData_(3,18)
{

  //register products
  produces<FEDRawDataCollection>();
  //if do put with a label
  //produces<FEDRawDataCollection>("someLabel");
  
  //now do what ever other initialization is needed
  eventPos_[0]=0; eventPos_[1]=0;
  eventCounter_=0; fileCounter_ = 0;
  run_ = 1; event_ = 1; bxn_ =1;
  isOpen_ = false; noMoreData_ = false; 
  debug_ = pset.getUntrackedParameter<bool>("PrintOut",false);
  triggerFedId_  = pset.getUntrackedParameter<unsigned int>("TriggerFedId",790);
  tbNum_ = pset.getUntrackedParameter<unsigned int>("TriggerBoardNum",1);
}

//Dtor
RPCFileReader::~RPCFileReader(){}

// ------------ Method called to set run & event numbers  ------------
void RPCFileReader::setRunAndEventInfo(){
    
  if(!isOpen_){
    if(fileCounter_<(int)fileNames().size()){
      eventPos_[0]=0; eventPos_[1]=0;
      isOpen_=true;
      fileCounter_++;
      edm::LogInfo("RPCFR")<< "[RPCFileReader::setRunAndEventInfo] "
			   << "Open for reading file no. " << fileCounter_+1
			   << " " << fileNames()[fileCounter_-1].substr(5);  
    }
    else{
      edm::LogInfo("RPCFR")<< "[RPCFileReader::setRunAndEventInfo] "
			   << "No more events to read. Finishing after " 
			   << eventCounter_ << " events.";
      noMoreData_=true;
      return;
    }
  }

  ReadDataFromAsciiFile(fileNames()[fileCounter_-1].substr(5),eventPos_);
  unsigned int freq = 100;
  if(debug_) freq = 1;
  if(eventCounter_%freq==0)
    edm::LogInfo("RPCFR") << "[RPCFileReader::setRunAndEventInfo] "
			  << " #" << eventCounter_ << " Run: "<< run_ 
			  << " Event: " << event_ << " Bx = " << bxn_ 
			  << " Time Stamp: " << timeStamp_.hour << ":" 
			  << timeStamp_.min << ":" << timeStamp_.sec
			  << " " << timeStamp_.day << " " << timeStamp_.month 
			  << " " << timeStamp_.year;

  setRunNumber(run_);
  setEventNumber(event_);
  //setTime(timeStamp_);
    
  return;
}

// ------------ Method called to produce the data  ------------
bool RPCFileReader::produce(edm::Event &ev) {
  std::auto_ptr<FEDRawDataCollection> result(new FEDRawDataCollection);
  
  //exit when no more data
  if(noMoreData_) return false;

  //fill the FEDRawDataCollection
  FEDRawData *rawData = RPCDataFormatter();
  FEDRawData& fedRawData = result->FEDData(triggerFedId_);
  fedRawData = *rawData;

  ev.put(result);

  return true;
}

// ------------ Method called to read the PAC data from ASCII file ------------
void RPCFileReader::ReadDataFromAsciiFile(string fileName="PacReadout.txt", int *pos=0){
  
  if(!pos){
    edm::LogError("RPCFR")<< "[RPCFileReader::ReadDataFromAsciiFile] "
			  << "Unknown event position; Abort";
    return;
  }

  ifstream infile;
  infile.open(fileName.c_str(),ios::binary);

  string dummy;
  int idummy;

  bool currentEvent=true;

  infile.seekg(pos[0]);
  
  while(infile >> dummy){
    if(dummy=="Beginning"){
      infile >> dummy;
      if(dummy=="Event"){
	if(!currentEvent) {
	  pos[0]=pos[1];
	  break;
	}
	infile >> dummy >> dec >> event_;
	infile >> dummy >> dec >> bxn_;
	infile >> dummy >> dummy >> dec >> timeStamp_.month >> timeStamp_.day;
	infile >> setw(2) >> dec >> timeStamp_.hour;
	infile.ignore(1);
	infile >> setw(2) >> dec >> timeStamp_.min;
	infile.ignore(1);
	infile >> setw(2) >> dec >> timeStamp_.sec;
	infile >> dec >> timeStamp_.year;
	currentEvent=false;
	eventCounter_++;
      }
      else if(dummy=="Run"){
	infile >> dummy >> dec >> run_;
      }
      else{
	if(debug_)
	  edm::LogInfo("RPCFR")<< "[RPCFileReader::ReadDataFromAsciiFile] "
				<< "Unrecognized header. Skipping ";
	infile.ignore(10000,10);// 10==\n
      }
    }
    else if(IsHexNumber(dummy)){
      int bxLocal = atoi(dummy.c_str());
      int bxn4b, bc0, valid;
      infile >> hex >>  bxn4b >> bc0 >> valid;
      if(valid==0x3ffff){ 
	for(int iL=0; iL<18; iL+=3){//Link data 
	  infile >> dummy;
	  for(int iLb=0;iLb<3;iLb++){
	    infile >> hex >> idummy;
	    RPCPacData dummyPartData(idummy);
	    if(dummyPartData.PartitionDelay()-bxLocal==RPC_PAC_L1ACCEPT_BX){//Link data collected at L1A bx
	      linkData_[dummyPartData.LbNum()][iL+iLb]=dummyPartData;
	    }
	  }
	}
	for(int iC=0; iC<12*3; iC+=3){//LogCones 
	  infile >> dummy;
	  LogCone logCone;
	  infile >> hex >> logCone.quality >> logCone.ptCode >> logCone.sign;
	  if(bxLocal==RPC_PAC_L1ACCEPT_BX+RPC_PAC_TRIGGER_DELAY){//write LogCones
	    theLogCones_[iC/3]=logCone;
	  }
	}
	infile >> dummy;
      }
      else{
	if(debug_)
	  edm::LogInfo("RPCFR")<< "[RPCFileReader::ReadDataFromAsciiFile] "
				<< " Not valid data. Validity bit = " << valid
				<< " Skipping record.";
	infile.ignore(10000,10);// 10==\n
      }
      pos[1]=infile.tellg();
    }
    else{
      if(debug_)
	edm::LogInfo("RPCFR")<< "[RPCFileReader::ReadDataFromAsciiFile] "
			      << "Unrecognized begining of record: " << dummy
			      << " Skipping record.";
      infile.ignore(10000,10);// 10==\n
    }
  }
  if(debug_)
    edm::LogInfo("RPCFR") << "[RPCFileReader::ReadDataFromAsciiFile] "
			   << " #" << eventCounter_ << " Run: "<< run_ 
			   << " Event: " << event_ << " Bx = " << bxn_ 
			   << " Time Stamp: " << timeStamp_.hour << ":" 
			   << timeStamp_.min << ":" << timeStamp_.sec
			   << " " << timeStamp_.day << " " << timeStamp_.month 
			   << " " << timeStamp_.year;
  if(infile.eof()){
    isOpen_=false;
  }
  else{
    isOpen_=true;
  }

  infile.close();

  return;
}

// ------------ Methods called to form RPC word for DAQ ------------
RPCFileReader::Word16 RPCFileReader::BuildCDWord(RPCPacData linkData){//Chamber Data(Link Board Data)
  Word16 word = (Word16(linkData.LbNum())<<14)
               |(Word16(linkData.PartitionNum())<<10)
               |(Word16(linkData.EndOfData())<<9)
               |(Word16(linkData.HalfPartition())<<8)
               |(Word16(linkData.PartitionData())<<0);
  return word;
}

RPCFileReader::Word16 RPCFileReader::BuildSLDWord(unsigned int tbNum, unsigned int linkNum){//Start of the Link input Data
  Word16 specIdent = 11;
  Word16 word = (specIdent<<14)
               |(1<<13)|(1<<12)|(1<<11)
               |(Word16(tbNum)<<5)
               |(Word16(linkNum)<<0);
  return word;
}

RPCFileReader::Word16 RPCFileReader::BuildSBXDWord(unsigned int bxn){//Start of the Bunch Crossing Data
  Word16 specIdent = 11;
  Word16 word = (specIdent<<14)
               |(0<<13)|(1<<12)
               |(Word16(bxn)<<0);
  return word;
}

RPCFileReader::Word16 RPCFileReader::BuildEmptyWord(){//Empty word
  Word16 specIdent = 11;
  Word16 word = (specIdent<<14)
               |(1<<13)|(0<<12)|(1<<11)
               |~(0x3ff);
  return word;
}

// ------------ Method called to format RPC Raw data  ------------
FEDRawData* RPCFileReader::RPCDataFormatter(){

  std::vector<Word16> words;
  bool empty=true;

  //Check if an event consists data
  for(unsigned int iL=0; iL<18; iL++){
    for(unsigned int iLb=0; iLb<2; iLb++){
      if(linkData_[iLb][iL].PartitionData()!=0)
	empty=false;
    }
  }
  if(!empty){
    //fill vector words with correctly ordered RPCwords
    words.push_back(BuildSBXDWord((unsigned int)bxn_));
    for(unsigned int iL=0; iL<18; iL++){
      //Check if data of current link exist
      empty=true;
      for(unsigned int iLb=0; iLb<2; iLb++){
	if(linkData_[iLb][iL].PartitionData()!=0)
	  empty=false;
      }
      if(!empty){
	words.push_back(BuildSLDWord(tbNum_, iL));//FIMXE iL+1??
	for(unsigned int iLb=0; iLb<2; iLb++){
	  if(linkData_[iLb][iL].PartitionData()!=0){
	    words.push_back(BuildCDWord(linkData_[iLb][iL]));
	  }
	}
      }
    }
  }
  //add empty words if needed
  while(words.size()%4!=0){
    words.push_back(BuildEmptyWord());
  }
  //format data, add header & trailer
  int dataSize = words.size()*sizeof(Word16);
  FEDRawData *rawData = new FEDRawData(dataSize);
  Word64 *word = reinterpret_cast<Word64*>(rawData->data());
  //FIXME
  //Add header
  //fedh_t *header = reinterpret_cast<fedh_t*>(word);
  //header->sourceid = (triggerFedId_<<8);
  //header->eventid = (0x50000000 | event_);
  //word++;//?
  //Add data
  for(unsigned int i=0; i<words.size(); i+=4){
    *word = (Word64(words[i])  <<48)
           |(Word64(words[i+1])<<32)
	   |(Word64(words[i+2])<<16)
           |(Word64(words[i+3])    );
    if(debug_){
      edm::LogInfo("RPCFR") << "[RPCFileReader::RPCDataFormater] Word64: " << *reinterpret_cast<bitset<64>* >(word);
    }
    word++;
  }
  //FIXME
  //Add trailer
  //fedt_t *trailer = reinterpret_cast<fedt_t*>(&word[(sizeof(fedh_t)+dataSize)/sizeof(Word64)]);
  //trailer->conscheck = 0xdeadface;
  //trailer->eventsize = (0xa0000000
  //			|((sizeof(fedh_t)+dataSize+sizeof(fedt_t))>>3));
  //check memory
  if(word!=reinterpret_cast<Word64*>(rawData->data()+dataSize)){
  //if(word!=reinterpret_cast<Word64*>(rawData->data()+sizeof(fedh_t)+dataSize+sizeof(fedt_t))){
    string warn = "ERROR !!! Problem with memory in RPCDataFormater !!!";
    throw cms::Exception(warn);
  }

  return rawData;
}
