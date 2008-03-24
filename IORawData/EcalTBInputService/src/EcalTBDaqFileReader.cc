#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "EcalTBDaqFileReader.h"
#include "EcalTBDaqSimpleFile.h"
#include "EcalTBDaqRFIOFile.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<iostream>
#include<vector>
#include<fstream>
#include <string>


using namespace std;
using namespace edm;

EcalTBDaqFileReader::EcalTBDaqFileReader(): initialized_(false), inputFile_(0) { 
  LogDebug("EcalTBInputService") << "@SUB=EcalTBDaqFileReader";

}


EcalTBDaqFileReader::~EcalTBDaqFileReader(){ 

  LogDebug("EcalTBInputService") << "@SUB=EcalTBDaqFileReader";

  if(inputFile_) {
    inputFile_->close();
    delete inputFile_;
    inputFile_=0;
  }

}

void EcalTBDaqFileReader::setInitialized(bool value){initialized_=value;}

bool EcalTBDaqFileReader::isInitialized(){
  
  return initialized_;

}



void EcalTBDaqFileReader::initialize(const string & file, bool isBinary){

  isBinary_ = isBinary;
  std::string protocol=file.substr( 0, file.find( ":" )) ;
  std::string filename=file.substr( file.find( ":" )+1, file.size());
  
  // case of ASCII input data file
  if (initialized_) {
    //    cout << "EcalTB DaqFileReader was already initialized... reinitializing it " << endl;
    if(inputFile_) {
      inputFile_->close();
      delete inputFile_;
      inputFile_=0;
    }
  }

  if (protocol == "file")
    inputFile_ = dynamic_cast<EcalTBDaqFile*>(new EcalTBDaqSimpleFile(filename, isBinary_));
  else if (protocol == "rfio")
    inputFile_ = dynamic_cast<EcalTBDaqFile*>(new EcalTBDaqRFIOFile(filename, isBinary_));

  // Initialize cachedData_
  cachedData_.len=0;
  cachedData_.fedData=0;
  initialized_ = true;
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

      if( inputFile_->checkEndOfFile() ) throw 1;
      
      // FedDataPair is struct: event lenght + pointer to beginning of event 
      if (!inputFile_->getEventData(cachedData_))
	throw 2;
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
	  return false;
	} 
      else 
	{
	  LogError("EcalTBInputService") << "@SUB=EcalTBDaqFileReader::fillDaqEventData" << "unkown exception";
	  return false;
	}
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

bool EcalTBDaqFileReader::checkEndOfFile() const
{ 
  return inputFile_->checkEndOfFile(); 
}

