#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EcalTBDaqSimpleFile.h"
#include <iosfwd>
#include <algorithm>
#include <iostream>

using namespace edm;

EcalTBDaqSimpleFile::EcalTBDaqSimpleFile(const std::string& filename, const bool& isBinary) : filename_(filename), isBinary_(isBinary)
{
  if ( isBinary_ )
    {
      // case of binary input data file
      LogInfo("EcalTBDaqSimpleFile") << "@SUB=EcalTBDaqSimpleFile::initialize" << "Opening binary data file " << filename;
      infile_.open( filename.c_str(), ios::binary );
      if( infile_.fail() ) 
	{
	  LogError("EcalTBDaqSimpleFile") << "@SUB=EcalTBDaqSimpleFile::initialize" << "the input file: " << filename << " is not present. Exiting program... " ;
	  exit(1);
	}
    }// end if binary
  
  else
    {
      LogInfo("EcalTBDaqSimpleFile") << "@SUB=EcalTBDaqSimpleFile::initialize" << "Opening ASCII file " << filename;
      infile_.open(filename.c_str());
      if( infile_.fail() ) 
	{
	  LogError("EcalTBDaqSimpleFile") << "@SUB=EcalTBDaqSimpleFile::initialize" << "the input file: " << filename << " is not present. Exiting program... " ;
	  exit(1);
	}
    }// end if not binary
}

//fill the event
bool EcalTBDaqSimpleFile::getEventData(FedDataPair& data) {

  if (isBinary_){
    ulong dataStore=1;
    
    // read first words of event, seeking for event size 
    infile_.read(reinterpret_cast<char *>(&dataStore),sizeof(ulong));
    infile_.read(reinterpret_cast<char *>(&dataStore),sizeof(ulong));
    infile_.read(reinterpret_cast<char *>(&dataStore),sizeof(ulong));
    int size =  dataStore & 0xffffff ;
//     cout << "[EcalTB DaqFileReader::getEventTrailer] event size masked  "<<  size 
// 	 << " (max size in byte is " << maxEventSizeInBytes_ << ")"<< endl;
    if (size > EcalTBDaqFile::maxEventSizeInBytes_){
      LogWarning("EcalTBDaqSimpleFile") << "@SUB=EcalTBDaqSimpleFile::getEventData" << "event size larger than allowed";
      return false;
    }
    
    infile_.seekg(-2*4,ios::cur);
    infile_.read(reinterpret_cast<char *>(&dataStore),sizeof(ulong));
    //cout << "is it beginning?\t"<< a << endl;
    infile_.seekg(-2*4,ios::cur);
    
    // 
    char   * bufferChar =  new  char [8* size];
    infile_.read (bufferChar,size*8);
    
    data.len         = size*8;
    data.fedData = reinterpret_cast<unsigned char*>(bufferChar);
    
  }// end in case of binary file
  

  else{
    string myWord;

    // allocate max memory allowed, use only what needed
    int len=0;
    ulong* buf = new ulong [EcalTBDaqFile::maxEventSizeInBytes_];
    ulong* tmp=buf;

    // importing data word by word
    for ( int i=0; i< EcalTBDaqFile::maxEventSizeInBytes_/4; ++i) {
      infile_ >> myWord;
      if (sscanf( myWord.c_str(),"%x",buf) == EOF)
	return false;
      int val= *buf >> 28;

//       if (i<4)
// 	cout << " myWord " << myWord << " myWord >> 28 " << val << endl;


      if ( len%2!=0 ) {
	// looking for end of event searching a tag
	if ( val  == EcalTBDaqFile::EOE_ ) {
	  //        cout << " EcalTBDaqSimpleFile::getEventTrailer EOE_ reached " <<   endl;
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

    data.len = len;
    //fedData = reinterpret_cast<unsigned char*>(tmp);
    data.fedData = reinterpret_cast<unsigned char*>(tmp);
    

  }// end in case of ASCII file

  return true;
}

//Check if the position in file is EOF
bool EcalTBDaqSimpleFile::checkEndOfFile()
{
  //unsigned char * buf = new unsigned char;
  long curr=infile_.tellg();
  infile_.seekg(0,ios::end);
  long end=infile_.tellg();
  infile_.seekg(curr,ios::beg);
  return (curr==end);
}

//Check if the position in file is EOF
void EcalTBDaqSimpleFile::close()
{
  infile_.close();
}
