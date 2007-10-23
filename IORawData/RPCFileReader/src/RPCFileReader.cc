
#include <stdexcept>
#include <memory>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <cctype>
#include <vector>
#include <bitset>
#include <ctime>

#include "IORawData/RPCFileReader/interface/RPCFileReader.h"
#include "IORawData/RPCFileReader/interface/RPCPacData.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"


using namespace std;
using namespace edm;

//Ctor
RPCFileReader::RPCFileReader(ParameterSet const& pset, 
			     InputSourceDescription const& desc):
  ExternalInputSource(pset, desc)
{

  //register products
  produces<FEDRawDataCollection>();
  produces<std::vector<L1MuRegionalCand> >("MTCCData");
  //if do put with a label
  //produces<FEDRawDataCollection>("someLabel");
  
  //now do what ever other initialization is needed
  eventPos_[0]=0; eventPos_[1]=0;
  eventCounter_=0; fileCounter_ = 0;
  run_ = 1; event_ = 1; bxn_ =1;
  isOpen_ = false; noMoreData_ = false; 

  debug_ = pset.getUntrackedParameter<bool>("PrintOut",false);
  saveOutOfTime_ = pset.getUntrackedParameter<bool>("SaveOutOfTimeDigis",false);
  pacTrigger_ = pset.getUntrackedParameter<bool>("IsPACTrigger",false);
  triggerFedId_  = pset.getUntrackedParameter<unsigned int>("TriggerFedId",790);
  const vector<unsigned int> tbNumdef(1,1);
  tbNums_ = pset.getUntrackedParameter<std::vector<unsigned int> >("TriggerBoardNums",tbNumdef);
  maskChannels_ = pset.getUntrackedParameter<bool>("MaskChannels",false);
  if(tbNums_.size()==0)tbNums_.push_back(1);

  //set vector sizes
  for(int k=0;k<(int)tbNums_.size();k++){
    vector<LogCone> lcv(12);
    theLogCones_.push_back(lcv);
    vector<RPCPacData> rpdv(3,RPCPacData(0));
    vector<vector<RPCPacData> > rpdvv(18,rpdv); 
    vector<vector<vector<RPCPacData> > > rpdvvv(LAST_BX-FIRST_BX,rpdvv); 
    linkData_.push_back(rpdvvv);
  }
  //define masked channels
  vector<bool> bv(18,0);
  for(unsigned int iChip=0;iChip<tbNums_.size();iChip++){
    maskedChannels.push_back(bv);
  }
  if(maskChannels_){//read masked channels from file FIXME 
    ifstream mfile;
    mfile.open("masks.dat");

    //check if file exists
    if((mfile.rdstate()&ifstream::failbit)!= 0){
      std::cerr << "Error opening masks.dat" << std::endl;
      edm::LogError("RPCFR")<< "[RPCFileReader::RPCFileReader] "
			    << "Error opening file with masks definition";
      mfile.close();
    }else{
      int chan;
      for(unsigned int iChip=0;iChip<tbNums_.size();iChip++){
	for(unsigned int iL=0;iL<18;iL++){
	  mfile >> chan;
	  maskedChannels[iChip][iL]=(bool)chan;
	}
      }
      mfile.close();
    }
    if(debug_){
      edm::LogInfo("RPCFR")<< "[RPCFileReader::RPCFileReader] Masked channels:";
      for(unsigned int iChip=0;iChip<tbNums_.size();iChip++){
	ostringstream ostr;
	ostr << iChip <<". (DCC channel = " << tbNums_[iChip]<<"): ";
	for(int iL=0;iL<18;iL++){
	  ostr<<(int)maskedChannels[iChip][iL]<<" ";
	}
	edm::LogInfo("RPCFR")<<ostr.str(); 
      }
    }
  } 
}

//Dtor
RPCFileReader::~RPCFileReader(){}

// ------------ Method called to set run & event numbers  ------------
void RPCFileReader::setRunAndEventInfo(){
    
 
  bool triggeredOrEmpty = false;
  tm aTime;

  while(!triggeredOrEmpty){
    if(!isOpen_){
      if(fileCounter_<(int)fileNames().size()){
	eventPos_[0]=0; eventPos_[1]=0;
	isOpen_=true;
	fileCounter_++;
	edm::LogInfo("RPCFR")<< "[RPCFileReader::setRunAndEventInfo] "
			     << "Open for reading file no. " << fileCounter_
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

    //clear vectors
    for(int iChip=0;iChip<(int)tbNums_.size();iChip++){
      for(int iBX=0;iBX<8;iBX++){
        for(int iL=0;iL<18;iL++){
	  for(int iLB=0;iLB<3;iLB++){
	    linkData_[iChip][iBX][iL][iLB] = RPCPacData();
	  }
        }
      }
    }
    
    readDataFromAsciiFile(fileNames()[fileCounter_-1].substr(5),eventPos_);

    bool isPacTrigger = false;
    for(int iChip=0; iChip<(int)tbNums_.size(); iChip++){
      for(int iCone=0; iCone<12; iCone++){
        isPacTrigger = (isPacTrigger||theLogCones_[iChip][iCone].ptCode);
      }
    }
    if(!(pacTrigger_)||isPacTrigger){
      //std::cout<<"Event triggered by PAC"<<std::endl;
      triggeredOrEmpty = true;
    }
    else{
      if(debug_)
	edm::LogInfo("RPCFR")<< "[RPCFileReader::setRunAndEventInfo] "
			     << " Data not triggered by PAC!";
      triggeredOrEmpty = false;
    }

    if(!isOpen_){//if eof don't write empty data into event
      triggeredOrEmpty = false;
    }

  }
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


  //Setting time stamp. To be optimised.
  int month = 0;
  if(timeStamp_.month=="Jan") month = 0;
  if(timeStamp_.month=="Feb") month = 1;
  if(timeStamp_.month=="March") month = 2;
  if(timeStamp_.month=="May") month = 3;
  if(timeStamp_.month=="April") month = 4;
  if(timeStamp_.month=="June") month = 5;
  if(timeStamp_.month=="July") month = 6;
  if(timeStamp_.month=="Aug") month = 7;
  if(timeStamp_.month=="Sep") month = 8;
  if(timeStamp_.month=="Nov") month = 9;
  if(timeStamp_.month=="Oct") month = 10;
  if(timeStamp_.month=="Dec") month = 11;
  aTime.tm_sec = timeStamp_.sec;
  aTime.tm_min = timeStamp_.min;
  aTime.tm_hour = timeStamp_.hour; //UTC check FIX!!
  aTime.tm_mday = timeStamp_.day;
  aTime.tm_mon = month;
  aTime.tm_year = timeStamp_.year - 1900;


  setRunNumber(run_);
  setEventNumber(event_);
  setTime(mktime(&aTime));  

  return;
}

// ------------ Method called to produce the data  ------------
bool RPCFileReader::produce(edm::Event &ev) {
  std::auto_ptr<FEDRawDataCollection> result(new FEDRawDataCollection);
  
  //exit when no more data
  if(noMoreData_) return false;

  unsigned int freq = 100;
  if(debug_) freq = 1;
  if(eventCounter_%freq==0)
    edm::LogInfo("RPCFR") << "[RPCFileReader::produce] "
			  << " #" << eventCounter_ << " Run: "<< run_ 
			  << " Event: " << event_ << " Bx = " << bxn_ 
			  << " Time Stamp: " << timeStamp_.hour << ":" 
			  << timeStamp_.min << ":" << timeStamp_.sec
			  << " " << timeStamp_.day << " " << timeStamp_.month 
			  << " " << timeStamp_.year;

  //fill the FEDRawDataCollection
  FEDRawData *rawData = rpcDataFormatter();
  FEDRawData& fedRawData = result->FEDData(triggerFedId_);
  fedRawData = *rawData;
  ev.put(result);

  //Trigger response
  float phi = 0;
  float eta = 0;
  std::vector<L1MuRegionalCand> RPCCand;
  for(unsigned int iChip=0;iChip<tbNums_.size();iChip++){
    for(unsigned int i=0;i<12;i++){
      if(!theLogCones_[iChip][i].ptCode) continue;
      L1MuRegionalCand l1Cand;        
      l1Cand.setQualityPacked(theLogCones_[iChip][i].quality);
      l1Cand.setPtPacked(theLogCones_[iChip][i].ptCode);    
      l1Cand.setType(0); 
      l1Cand.setChargePacked(1);
      l1Cand.setPhiValue(phi);
      l1Cand.setEtaValue(eta);
      RPCCand.push_back(l1Cand);
    }
  }

  std::auto_ptr<std::vector<L1MuRegionalCand> > candBarell(new std::vector<L1MuRegionalCand>);
  candBarell->insert(candBarell->end(), RPCCand.begin(), RPCCand.end());
  ev.put(candBarell, "MTCCData");
  
  return true;
}

// ------------ Method called to read the PAC data from ASCII file ------------
void RPCFileReader::readDataFromAsciiFile(string fileName, int *pos){
  
  if(!pos){
    edm::LogError("RPCFR")<< "[RPCFileReader::readDataFromAsciiFile] "
			  << "Unknown event position; Abort";
    return;
  }

  ifstream infile;
  infile.open(fileName.c_str(),ios::binary);

  //check if file exists
  if((infile.rdstate()&ifstream::failbit)!= 0){
    std::cerr << "Error opening " << fileName << std::endl;
    edm::LogError("RPCFR")<< "[RPCFileReader::readDataFromAsciiFile] "
			  << "Error opening data file " << fileName;
    isOpen_=false;
    infile.close();

    return;
  }

  string dummy;
  int idummy;
  int iChip=0;

  bool currentEvent=true;

  infile.seekg(pos[0]);
  
  while(infile >> setw(0)>> dummy){
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
      else if(dummy=="Header"){
	infile >> dummy >> dec >> run_;
      }
      else{
	if(debug_)
	  edm::LogInfo("RPCFR")<< "[RPCFileReader::readDataFromAsciiFile] "
			       << "Unrecognized header: "<< dummy
			       << " Skipping ";
	infile.ignore(10000,10);// 10==\n
      }
    }
    else if((dummy=="PAC")||(dummy=="RMB")){
      infile >> iChip;
      if(debug_)
	edm::LogInfo("RPCFR")<< "[RPCFileReader::readDataFromAsciiFile] "
			     << "Readout from "<< dummy << " " << iChip;	
    }
    else if(isHexNumber(dummy)){
      int bxLocal = atoi(dummy.c_str());
      int bxn4b, bc0, valid;
      const int valid_mask[18] = {0x20000,0x10000,0x8000,0x4000,
				  0x2000, 0x1000, 0x800, 0x400,
				  0x200,  0x100,  0x80,  0x40,
				  0x20,   0x10,   0x8 ,  0x4,
				  0x2,    0x1};
      infile >> hex >>  bxn4b >> bc0 >> valid;
      for(int iL=17; iL>=0; iL-=3){//Link data 
	infile >> dummy;
	for(int iLb=0;iLb>-3;iLb--){
	  infile >> hex >> idummy;
	  RPCPacData dummyPartData;
	  if(((valid&valid_mask[17-(iL+iLb)])>>(iL+iLb))==1){
	    dummyPartData.fromRaw(idummy);
	    //std::cout<<"VALID"<<std::endl;
	  }
	  else{
	    if(debug_)
	      edm::LogInfo("RPCFR")<< "[RPCFileReader::readDataFromAsciiFile] "
				   << " Not valid data for link no. = " << iL+iLb
				   << " Zeroing partition data.";
	  }
	  for(int iBX=FIRST_BX;iBX<LAST_BX;iBX++){	     
	    if(bxLocal-(int)dummyPartData.partitionDelay()==RPC_PAC_L1ACCEPT_BX+iBX){//Link data collected at L1A bx
	      linkData_[iChip][iBX-FIRST_BX][iL+iLb][dummyPartData.lbNum()]=dummyPartData;	    	
	    }
	  }
	}
      }
      for(int iC=12*3-1; iC>=0; iC-=3){//LogCones
	infile >> dummy;
	LogCone logCone;
	infile >> hex >> logCone.quality >> logCone.ptCode >> logCone.sign;
	if(logCone.ptCode){
	  edm::LogInfo("Any time LogCone")<<"Log cone: Q: "<<logCone.quality
					  <<" ptCode: "<<logCone.ptCode
					  <<" sign: "<<logCone.sign<<std::endl;
	}
	if(bxLocal==RPC_PAC_L1ACCEPT_BX+RPC_PAC_TRIGGER_DELAY){//write LogCones
	  theLogCones_[iChip][iC/3]=logCone;
	  if(logCone.ptCode){
	    edm::LogInfo("In time LogCone")<<"in time Log cone: Q: "<<logCone.quality
					   <<" ptCode: "<<logCone.ptCode
					   <<" sign: "<<logCone.sign<<std::endl;
	  }
	}
      }
      infile >> dummy;
      pos[1]=infile.tellg();
    }
    else{
      if(debug_)
	edm::LogInfo("RPCFR")<< "[RPCFileReader::readDataFromAsciiFile] "
			      << "Unrecognized begining of record: " << dummy
			      << " Skipping record.";
      infile.ignore(10000,10);// 10==\n
    }
  }
  if(debug_)
    edm::LogInfo("RPCFR") << "[RPCFileReader::readDataFromAsciiFile] "
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
RPCFileReader::Word16 RPCFileReader::buildCDWord(RPCPacData linkData){//Chamber Data(Link Board Data)

  Word16 word = (Word16(linkData.lbNum())<<14)
               |(Word16(linkData.partitionNum())<<10)
               |(Word16(linkData.endOfData())<<9)
               |(Word16(linkData.halfPartition())<<8)
               |(Word16(linkData.partitionData())<<0);
  if(debug_){
    edm::LogInfo("RPCFR") << "[RPCFileReader::buildCDWord] ";
  }
  return word;
}

RPCFileReader::Word16 RPCFileReader::buildSLDWord(unsigned int tbNum, unsigned int linkNum){//Start of the Link input Data

  Word16 specIdent = 3;
  Word16 word = (specIdent<<14)
               |(1<<13)|(1<<12)|(1<<11)
               |(Word16(tbNum)<<5)
               |(Word16(linkNum)<<0);
  if(debug_){
    edm::LogInfo("RPCFR") << "[RPCFileReader::buildSLDWord] ";
  }
  return word;
}

RPCFileReader::Word16 RPCFileReader::buildSBXDWord(unsigned int bxn){//Start of the Bunch Crossing Data
  Word16 specIdent = 3;
  
  Word16 word = (specIdent<<14)
               |(0<<13)|(1<<12)
               |(Word16(bxn)<<0);
    if(debug_){
    edm::LogInfo("RPCFR") << "[RPCFileReader::buildSBXDWord] ";
  }
  return word;
}

RPCFileReader::Word16 RPCFileReader::buildEmptyWord(){//Empty word
  Word16 specIdent = 3;
  Word16 word = 0;
  word = (specIdent<<14)
        |(1<<13)|(0<<12)|(1<<11);
  if(debug_){
    edm::LogInfo("RPCFR") << "[RPCFileReader::buildEmptyWord] ";
  }
  return word;
}

// ------------ Method called to format RPC Raw data  ------------
FEDRawData* RPCFileReader::rpcDataFormatter(){

  std::vector<Word16> words;
  bool empty=true;

  int beginBX = 0;
  int endBX = 1;
  if(saveOutOfTime_){
    //beginBX = -1;
    //endBX = 2;
    beginBX = FIRST_BX;
    endBX = LAST_BX;
  }
  for(unsigned int iChip=0;iChip<tbNums_.size();iChip++){
    for(int iBX=beginBX;iBX<endBX;iBX++){
      //Check if an event consists data
      for(unsigned int iL=0; iL<18; iL++){
	for(unsigned int iLb=0; iLb<3; iLb++){
	  if((linkData_[iChip][iBX-FIRST_BX][iL][iLb].partitionData()!=0)&&
	     !(maskedChannels[iChip][iL]))
	    empty=false;
	}
      }
      if(!empty){
	//fill vector words with correctly ordered RPCwords
	if(bxn_+iBX>=0)
	  words.push_back(buildSBXDWord((unsigned int)(bxn_+iBX)));
	else
	  continue;
	for(unsigned int iL=0; iL<18; iL++){
	  //Check if data of current link exist
	  empty=true;
	  for(unsigned int iLb=0; iLb<3; iLb++){
	    if((linkData_[iChip][iBX-FIRST_BX][iL][iLb].partitionData()!=0)&&
	       !(maskedChannels[iChip][iL]))
	      empty=false;
	  }
	  if(!empty){
	    words.push_back(buildSLDWord(tbNums_[iChip], iL));//FIMXE iL+1??
	    for(unsigned int iLb=0; iLb<3; iLb++){
	      if(linkData_[iChip][iBX-FIRST_BX][iL][iLb].partitionData()!=0){
		words.push_back(buildCDWord(linkData_[iChip][iBX-FIRST_BX][iL][iLb]));
	      }
	    }
	  }
	}
      }
      //Add empty words if needed
      while(words.size()%4!=0){
	words.push_back(buildEmptyWord());
      }
    }
  }

  if(debug_){
    edm::LogInfo("RPCFR") << "[RPCFileReader::rpcDataFormater] Num of words: " << words.size();
  }
  //Format data, add header & trailer
  int dataSize = words.size()*sizeof(Word16);
  FEDRawData *rawData = new FEDRawData(dataSize+2*sizeof(Word64));
  Word64 *word = reinterpret_cast<Word64*>(rawData->data());
  //Add simple header by hand
  *word = Word64(0);
  *word = (Word64(0x5)<<60)|(Word64(0x3)<<56)|(Word64(event_)<<32)
         |(Word64(bxn_)<<20)|((triggerFedId_)<<8);
  if(debug_){
    edm::LogInfo("RPCFR") << "[RPCFileReader::rpcDataFormater] Header: " << *reinterpret_cast<bitset<64>* >(word);
  }
  word++;
  //Add data
  for(unsigned int i=0; i<words.size(); i+=4){
    *word = (Word64(words[i])  <<48)
           |(Word64(words[i+1])<<32)
	 |(Word64(words[i+2])<<16)
           |(Word64(words[i+3])    );
    if(debug_){
      edm::LogInfo("RPCFR") << "[RPCFileReader::rpcDataFormater] Word64: " << *reinterpret_cast<bitset<64>* >(word);
    }
    word++;
  }
  //Add simple trailer by hand
  *word = Word64(0);
  *word = (Word64(0xa)<<60)|(Word64(0x0)<<56)|(Word64(2+words.size()/4)<<32)
         |(0xf<<8)|(0xf<<4);
  if(debug_){
    edm::LogInfo("RPCFR") << "[RPCFileReader::rpcDataFormater] Trailer: " << *reinterpret_cast<bitset<64>* >(word);
  }
  word++;

  //Check memory
  if(word!=reinterpret_cast<Word64*>(rawData->data()+dataSize+2*sizeof(Word64))){
    string warn = "ERROR !!! Problem with memory in RPCFileReader::rpcDataFormater !!!";
    throw cms::Exception(warn);
    }

  return rawData;
}
