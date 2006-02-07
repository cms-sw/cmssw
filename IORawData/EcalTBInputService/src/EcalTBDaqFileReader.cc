#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "EcalTBDaqFileReader.h"


#include<iostream>
#include<vector>
#include<fstream>
#include <string>

using namespace std;
using namespace edm;
//using namespace raw;

EcalTBDaqFileReader * EcalTBDaqFileReader::instance_=0;

EcalTBDaqFileReader * EcalTBDaqFileReader::instance(){

  if (instance_== 0) {
    instance_ = new EcalTBDaqFileReader();
  }
  return instance_;

}


EcalTBDaqFileReader::EcalTBDaqFileReader(): initialized_(false) { 

  LogDebug("EcalTBInputService") << "@SUB=EcalTBDaqFileReader";
  if(instance_ == 0) instance_ = this;

}


EcalTBDaqFileReader::~EcalTBDaqFileReader(){ 

  LogDebug("EcalTBInputService") << "@SUB=EcalTBDaqFileReader";

  if(inputFile) {
    inputFile.close();
    //delete input_;
  }
  instance_=0;

  //Cleaning the event
  if (cachedData_.fedData)
    delete[] cachedData_.fedData;
}

void EcalTBDaqFileReader::setInitialized(bool value){initialized_=value;}

bool EcalTBDaqFileReader::isInitialized(){
  
  return initialized_;

}



void EcalTBDaqFileReader::initialize(const string & filename, bool isBinary){
  
  isBinary_ = isBinary;

  // case of ASCII input data file
  if (initialized_) {
    //    cout << "EcalTB DaqFileReader was already initialized... reinitializing it " << endl;
    if(inputFile) {
      inputFile.close();
    }
  }
  
  if ( isBinary_){
    // case of binary input data file
    LogInfo("EcalTBInputService") << "@SUB=EcalTBDaqFileReader::initialize" << "Opening binary data file " << filename;
    inputFile.open(filename.c_str(), ios::binary );
    if( inputFile.fail() ) {
      LogError("EcalTBInputService") << "@SUB=EcalTBDaqFileReader::initialize" << "the input file: " << filename << " is not present. Exiting program... " ;
      exit(1);
    }
  }// end if binary
  
  else{
    LogInfo("EcalTBInputService") << "@SUB=EcalTBDaqFileReader::initialize" << "Opening ASCII file " << filename;
    inputFile.open(filename.c_str());
    if( inputFile.fail() ) {
      LogError("EcalTBInputService") << "@SUB=EcalTBDaqFileReader::initialize" << "the input file: " << filename << " is not present. Exiting program... " ;
      exit(1);
    }
  }// end if not binary

  // Initialize cachedData_
  cachedData_.len=0;
  cachedData_.fedData=0;
  initialized_ = true;
}





//Fill the event
void EcalTBDaqFileReader::getEventTrailer() {
  
  if (isBinary_){
    ulong dataStore=1;
    
    // read first words of event, seeking for event size 
    inputFile.read(reinterpret_cast<char *>(&dataStore),sizeof(ulong));
    inputFile.read(reinterpret_cast<char *>(&dataStore),sizeof(ulong));
    inputFile.read(reinterpret_cast<char *>(&dataStore),sizeof(ulong));
    int size =  dataStore & 0xffffff ;
//     cout << "[EcalTB DaqFileReader::getEventTrailer] event size masked  "<<  size 
// 	 << " (max size in byte is " << maxEventSizeInBytes_ << ")"<< endl;
    if (size > maxEventSizeInBytes_){
      LogWarning("EcalTBInputService") << "@SUB=EcalTBDaqFileReader::getEventTrailer" << "event size larger than allowed";
    }
    
    inputFile.seekg(-2*4,ios::cur);
    inputFile.read(reinterpret_cast<char *>(&dataStore),sizeof(ulong));
    //cout << "is it beginning?\t"<< a << endl;
    inputFile.seekg(-2*4,ios::cur);
    
    // 
    char   * bufferChar =  new  char [8* size];
    inputFile.read (bufferChar,size*8);
    
    cachedData_.len         = size*8;
    cachedData_.fedData = reinterpret_cast<unsigned char*>(bufferChar);
    
  }// end in case of binary file
  

  else{
    string myWord;

    // allocate max memory allowed, use only what needed
    int len=0;
    ulong* buf = new ulong [maxEventSizeInBytes_];
    ulong* tmp=buf;

    // importing data word by word
    for ( int i=0; i< maxEventSizeInBytes_/4; ++i) {
      inputFile >> myWord;
      sscanf( myWord.c_str(),"%x",buf);
      int val= *buf >> 28;

//       if (i<4)
// 	cout << " myWord " << myWord << " myWord >> 28 " << val << endl;


      if ( len%2!=0 ) {
	// looking for end of event searching a tag
	if ( val  == EOE_ ) {
	  //        cout << " EcalTBDaqFileReader::getEventTrailer EOE_ reached " <<   endl;
	  len++;
	  break;
	}
      }
      buf+=1;
      len++;

    }// end loop importing data word by word

    //    cout << " Number of 32 words in the event " << len << endl;
    len*=4;
    //    cout << " Length in bytes " << len << endl;

    cachedData_.len = len;
    //fedData = reinterpret_cast<unsigned char*>(tmp);
    cachedData_.fedData = reinterpret_cast<unsigned char*>(tmp);
    

  }// end in case of ASCII file

}






bool EcalTBDaqFileReader::fillDaqEventData() {
  //  cout<< "EcalTBDaqFileReader::fillDaqEventData() beginning " << endl;      
  const int MAXFEDID = 1024;

  //Cleaning the event before filling
  if (cachedData_.fedData)
    delete[] cachedData_.fedData;

  pair<int,int> fedInfo;  //int =FED id, int = event data length 

  try 
    {
      
      //checkEndOfEvent()replaced by tru + getEventTrailer();
      
      if( checkEndOfFile() ) throw 1;
      
      // FedDataPair is struct: event lenght + pointer to beginning of event 
      getEventTrailer();
      // extracting header information from event
      setFEDHeader();
      
      fedInfo.first=getFedId();
      fedInfo.second = cachedData_.len;
      
//       cout << " fillDaqEventData Fed Id " << fedInfo.first << " getEventLength() " 
// 	   << getEventLength() << " run " << getRunNumber() << " Ev "
// 	   << getEventNumber() << endl;
      
      
      //     EventID ev( getRunNumber(), getEventNumber() );
      //     cID=ev;
      
      if(fedInfo.first<0)
	{
	  LogError("EcalTBInputService") << "@SUB=EcalTBDaqFileReader::addFEDRawData" << "negative FED Id. Adding no data";
	  throw 2;
	} 
      else if (fedInfo.first>MAXFEDID)
	{
	  LogError("EcalTBInputService") << "@SUB=EcalTBDaqFileReader::addFEDRawData" << "FED Id(" << fedInfo.first << ") greater than maximum allowed (" << MAXFEDID << "). Adding no data";
	  throw 3;
	} 
      return true;
    } 
  catch(int i) 
    {
      if (i==1)
	{
	  LogInfo("EcalTBInputService") << "@SUB=EcalTBDaqFileReader::fillDaqEventData" << "END OF FILE REACHED. No information read for the requested event";
	} 
      else 
	{
	  LogError("EcalTBInputService") << "@SUB=EcalTBDaqFileReader::fillDaqEventData" << "unkown exception";
	}
      return false;
    }
}


void  EcalTBDaqFileReader::setFEDHeader() {

//   cout<<"getting FED Header "<<  endl;
  int headsize=16;
  headValues_.clear();

  unsigned long* buf = reinterpret_cast<unsigned long*>(cachedData_.fedData);
  //  int val=0;
  
  for ( int i=0; i< headsize/4; ++i) {
    //    cout << i << " " << hex << buf << dec << endl ;
    
    if ( i==0) {    
      headValues_.push_back((*buf>>8)&0xFFF);   // DCC id    
      
    } else if ( i==1) {
      headValues_.push_back( (*buf)&0xFFFFFF);      // Lv1 number
      //      cout << " LV1  " << ((*buf)&0xFFFFFF) << endl;
    } else if ( i==2) {
      headValues_.push_back( ((*buf)&0xFFFFFF)*8 ); // Event length
      //      cout << " Event length " << ((*buf)&0xFFFFFF)*8 << endl;
    } else if ( i==3) {
      //      int runN= (*buf)&0xFFFFFF;
      //      cout << " runN " << runN << endl; 
      headValues_.push_back( (*buf)&0xFFFFFF); // Run NUmber
      
    }
    buf+=1;
  }

}


bool EcalTBDaqFileReader::checkEndOfFile() {

  //  cout << "  EcalTBDaqFileReader::checkEndOfFile " << endl;
  //unsigned char * buf = new unsigned char;

  bool retval=false;
  if ( inputFile.eof() ) retval=true;
  return retval;
}

