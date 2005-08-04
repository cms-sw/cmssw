#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EcalTBDaqFileReader.h"


#include<iostream>
#include<vector>
#include<fstream>
#include <string>

using namespace std;
using namespace edm;
using namespace raw;

EcalTBDaqFileReader * EcalTBDaqFileReader::instance_=0;

EcalTBDaqFileReader * EcalTBDaqFileReader::instance(){

  if (instance_== 0) {
    instance_ = new EcalTBDaqFileReader();
  }
  return instance_;

}


EcalTBDaqFileReader::EcalTBDaqFileReader(): initialized_(false) { 

  cout << "Constructing a new EcalTBDaqFileReader" << endl;
  if(instance_ == 0) instance_ = this;

}


EcalTBDaqFileReader::~EcalTBDaqFileReader(){ 

  cout << "Destructing EcalTBDaqFileReader" << endl;

 if(inputFile) {
    inputFile.close();
    //delete input_;
  }
  instance_=0;
}

void EcalTBDaqFileReader::setInitialized(bool value){initialized_=value;}

bool EcalTBDaqFileReader::isInitialized(){
  
  return initialized_;

}



void EcalTBDaqFileReader::initialize(const string & filename){


  if (initialized_) {
    cout << "EcalTB DaqFileReader was already initialized... reinitializing it " << endl;
    if(inputFile) {
      inputFile.close();
      //delete input_;
    }

  }

  cout<<"EcalTB DaqFileReader::initialize : Opening file "<<filename<<endl;
  inputFile.open(filename.c_str());
  if( inputFile.fail() ) {
    cout<<"EcalTBDaqFileReader: the input file: "<<filename <<" is not present. Exiting program... "  << endl;
    exit(1);
  }

  initialized_ = true;
}

FedDataPair 
EcalTBDaqFileReader::getEventTrailer() {

  //  cout << " EcalTBDaqFileReader getEvent Trailer  " << endl;
  string myWord;

  int len=0;
  ulong* buf = new ulong [maxEventSizeInBytes_];
  ulong* tmp=buf;

  for ( int i=0; i< maxEventSizeInBytes_/4; ++i) {
    inputFile >> myWord;
    sscanf( myWord.c_str(),"%x",buf);
    int val= *buf >> 28;
    //    cout << " myWord " << myWord << " myWord >> 28 " << val << endl;

    if ( len%2!=0 ) {

      if ( val  == EOE_ ) {
	//        cout << " EcalTBDaqFileReader::getEventTrailer EOE_ reached " <<   endl;
        len++;
        break;
      }
    }
    buf+=1;
    len++;

  }

  //  cout << " Number of 32 words in the event " << len << endl;
  len*=4;

  FedDataPair aPair;
  aPair.len = len;
  //aPair.fedData = fedData;
  //fedData = reinterpret_cast<unsigned char*>(tmp);
  aPair.fedData = reinterpret_cast<unsigned char*>(tmp);

  for ( int i=0; i<20; ++i) {
    //    cout << "i)"<< i << " " << hex<<(*fedData)<< endl; 
    //fedData++;
  }

  return aPair;
}


bool EcalTBDaqFileReader::fillDaqEventData(edm::CollisionID & cID, FEDRawDataCollection& data ) {
  //  cout<< "EcalTBDaqFileReader::fillDaqEventData() beginning " << endl;      
  const int MAXFEDID = 1024;

  pair<int,int> fedInfo;  //int =FED id, int = event data length 
  fedInfo.first=1;  //FAKE

  int fedId=fedInfo.first;

  try {

    //checkEndOfEvent()replaced by tru + getEventTrailer();

    if( checkEndOfFile() ) throw 1;

    FedDataPair aPair = getEventTrailer();
    fedInfo.second = aPair.len;

    if(fedId<0){
      cerr<<"DaqEvent::addFEDRawData - ERROR : negative FED Id. Adding no data"<<endl;
    } else if (fedId>MAXFEDID){
      cerr<<"DaqEvent::addFEDRawData - ERROR : FED Id("<<fedId<<") greater than maximum allowed ("<<MAXFEDID<<"). Adding no data"<<endl;
    }  else {

      //if ( infoV ) cout<<"DaqEvent: adding DaqFEDRawData of FED"<< fedId << endl;
      cout<<"DaqEvent: adding DaqFEDRawData of FED"<< fedId << endl;

      FEDRawData& eventfeddata = data.FEDData( fedInfo.first  );
      eventfeddata.data_.resize(fedInfo.second);
      copy(aPair.fedData, aPair.fedData + fedInfo.second , eventfeddata.data_.begin());
      cout<< "EcalTBDaqFileReader::fillDaqEventData() : read in full event information" << endl;
    }
    // ShR 4Aug05: this is nasty but needed to fix big memory leak in current implementation
    delete[] aPair.fedData;
    return true;
  } catch(int i) {
    if (i==1){
      cout<<"END OF FILE REACHED. No information read for the requested event"<<endl;
    } else {
      cout<< "unkown exception in EcalTBDaqFileReader::fillDaqEventData()" << endl;
    }
    return false;
  }
}

bool EcalTBDaqFileReader::checkEndOfFile() {

  //  cout << "  EcalTBDaqFileReader::checkEndOfFile " << endl;
  //unsigned char * buf = new unsigned char;

  bool retval=false;
  if ( inputFile.eof() ) retval=true;
  return retval;
}

bool EcalTBDaqFileReader::checkEndOfEvent() {

  //getEventTrailer();
  cout << "EcalTBDaqFileReader::checkEndOfEvent() not implemented! returning dummy true value!"
       << endl;
  return true;
}
