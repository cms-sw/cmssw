#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EcalTBDaqRFIOFile.h"

#include <sys/types.h>
#include <sys/fcntl.h>
#include <stdio.h>

#include <iosfwd>
#include <algorithm>
#include <iostream>

extern "C" {
  extern int rfio_open  (const char *path, int flags, int mode);
  extern FILE *rfio_fopen (char *path, char *mode);
  extern int rfio_fread(void*, size_t, size_t, void*);
  extern int rfio_fclose (FILE *fd);
  extern int rfio_fseek (FILE *fp, long offset, int whence);
  extern int rfio_feof (FILE *fp);
  extern long rfio_ftell (FILE *fp);
}

using namespace edm;

EcalTBDaqRFIOFile::EcalTBDaqRFIOFile(const std::string& filename, const bool& isBinary) : filename_(filename), isBinary_(isBinary)
{
  if ( isBinary_ )
    {
      // case of binary input data file
      LogInfo("EcalTBDaqRFIOFile") << "@SUB=EcalTBDaqRFIOFile::initialize" << "Opening binary data file " << filename;
      int fd = rfio_open(filename.c_str(), O_RDONLY, 0644);
      if( fd < 0 ) 
	{
	  LogError("EcalTBDaqRFIOFile") << "@SUB=EcalTBDaqRFIOFile::initialize" << "the input file: " << filename << " cannot be opened. Exiting program... " ;
	  exit(1);
	}
      infile_ = rfio_fopen((char*)filename.c_str(), "r");
    }// end if binary

  else
    {
      LogInfo("EcalTBDaqRFIOFile") << "@SUB=EcalTBDaqRFIOFile::initialize" << "Opening ASCII file " << filename;
      int fd = rfio_open(filename.c_str(), O_RDONLY, 0644);
      if( fd < 0 )
        {
          LogError("EcalTBDaqRFIOFile") << "@SUB=EcalTBDaqRFIOFile::initialize" << "the input file: " << filename << " cannot be opened. Exiting program... " ;
          exit(1);
        }
      infile_ = rfio_fopen((char*)filename.c_str(), "r");
    }// end if not binary
}

//fill the event
bool EcalTBDaqRFIOFile::getEventData(FedDataPair& data) {

  if (isBinary_){
    ulong dataStore=1;
  
    // read first words of event, seeking for event size 
    if (rfio_fread(reinterpret_cast<char *>(&dataStore),sizeof(ulong),1,infile_) < 0)
      return false;
    if (rfio_fread(reinterpret_cast<char *>(&dataStore),sizeof(ulong),1,infile_) < 0)
      return false;
    if (rfio_fread(reinterpret_cast<char *>(&dataStore),sizeof(ulong),1,infile_) < 0)
      return false;

    int size =  dataStore & 0xffffff ;
    //     cout << "[EcalTB DaqFileReader::getEventTrailer] event size masked  "<<  size 
    // 	 << " (max size in byte is " << maxEventSizeInBytes_ << ")"<< endl;
    if (size > EcalTBDaqFile::maxEventSizeInBytes_){
      LogWarning("EcalTBDaqRFIOFile") << "@SUB=EcalTBDaqRFIOFile::getEventData" << "event size larger than allowed";
      return false;
    }
  

    if (rfio_fseek(infile_,-2*4,SEEK_CUR) < 0)
      return false ;
    if (rfio_fread(reinterpret_cast<char *>(&dataStore),sizeof(ulong),1,infile_) < 0)
      return false;
    //cout << "is it beginning?\t"<< a << endl;
    if (rfio_fseek(infile_,-2*4,SEEK_CUR) < 0)
      return false;
  
    // 
    char   * bufferChar =  new  char [8* size];
    if (rfio_fread(bufferChar,size*8,1,infile_) < 0)
      return false;

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
//      infile_ >> myWord;
/**/
      myWord = "";
      while ( 1 ) {
        char c[1];
        if (rfio_fread(c,1,1,infile_) < 0)
          return false;
        if ( c[0] == '\n' )
          break;
        myWord += c[0];
      }
/**/
      if (sscanf( myWord.c_str(),"%x",buf) == EOF)
        return false;
      int val= *buf >> 28;

//       if (i<4)
//      cout << " myWord " << myWord << " myWord >> 28 " << val << endl;


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
bool EcalTBDaqRFIOFile::checkEndOfFile()
{
  //unsigned char * buf = new unsigned char;
  long curr=rfio_ftell(infile_);
  rfio_fseek(infile_,0,SEEK_END);
  long end=rfio_ftell(infile_);
  rfio_fseek(infile_,curr,SEEK_SET);
  return (curr==end);
}

//Check if the position in file is EOF
void EcalTBDaqRFIOFile::close()
{
  if (infile_) {
    rfio_fclose(infile_);
    infile_ = 0;
  }
}
