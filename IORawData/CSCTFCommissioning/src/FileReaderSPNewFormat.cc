#include <IORawData/CSCTFCommissioning/src/FileReaderSPNewFormat.h>

#include <iostream>
#include <fstream>
#include <cstdio>
#include <unistd.h>
#include <cstring>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

FileReaderSPNewFormat::FileReaderSPNewFormat()
{
  pointer = 0;
  trailer[0] = 0xf000;
  trailer[1] = 0xf07f;
  trailer[2] = 0xf000;
  trailer[3] = 0xf000;
  trailer[4] = 0xe000;
  trailer[5] = 0xe000;
  trailer[6] = 0xe000;
  trailer[7] = 0xe000;

  header[0] = 0x9000;
  header[1] = 0x9000;
  header[2] = 0x9000;
  header[3] = 0x9000;
  header[4] = 0xa000;
  header[5] = 0xa000;
  header[6] = 0xa000;
  header[7] = 0xa000;
}

void FileReaderSPNewFormat::Configure() {
}


int FileReaderSPNewFormat::chunkSize() {
  // The smallest piece of the data format 
  // is 64 bits so read in that amount.
  return sizeof(long long);
}


int FileReaderSPNewFormat::readSP(unsigned short **buf, const bool debug) 
{
  unsigned short a[50000];
  int bytes_read=0;
  unsigned trailer_pos;
  unsigned counter = 0;

  do
    {
      if(pointer == 0)
	{
	  bytes_read = read(fd_schar, a, chunkSize());  //read 1 chunk
	  counter += bytes_read;
	  pointer += counter;
	  if(bytes_read != chunkSize())
	    {
	      edm::LogInfo("FileReaderSPNewFormat") << "Reached end of file.";
	      return 0; // we can safely return zero since an event size less than one chunk is useless.
	    }	  
	}
      else
	{
	  bytes_read = read(fd_schar, a + counter/sizeof(short), chunkSize());
	  counter += bytes_read;
	  pointer += counter;
	  if(bytes_read != chunkSize())
	    {
	      edm::LogInfo("FileReaderSPNewFormat") << "Reached end of file, recovering last data.";
	      if( !findTrailer2(a, counter/sizeof(short)) )
		{
		  if( (trailer_pos = findTrailer1(a, counter)) ) // if the first trailer is there
		    {
		      memcpy(a + trailer_pos + 4, trailer + 4, 4*sizeof(short));
		      counter += 4;
		    }
		  else
		    {
		      memcpy(a + counter/sizeof(short), trailer, 8*sizeof(short));
		      counter += 8;
		    }
		}
	    }
	}
      
    }while( !hasTrailer(a, counter/sizeof(short)) );
  
  void * event = malloc(counter);
  memset(event,0,counter);
  memcpy(event,a,counter);
  *buf=reinterpret_cast<unsigned short*>(event);
    
  //printf("\n\n\n  returning %d: \n\n\n",(count*8));

  return counter;  //Total number of bytes read

} //readSP

unsigned FileReaderSPNewFormat::findTrailer1(const unsigned short* buf, unsigned length)
{
  unsigned short* itr = const_cast<unsigned short*>(buf);

  unsigned pos = 0;
   
  while( itr + 3 <= buf + length )
    {
      //      if(isTrailer1(itr)) pos = (unsigned)itr; 
      // Cannot reinterpret to unsigned int directly since it claims loss of precision.
      // But it accepts silent cast from unsigned long to unsigned int in assignment
      if(isTrailer1(itr)) pos = reinterpret_cast<unsigned long>(itr); 
      ++itr;
    }
  
  return pos;
}

unsigned FileReaderSPNewFormat::findTrailer2(const unsigned short* buf, unsigned length)
{
  unsigned short* itr = const_cast<unsigned short*>(buf);

  unsigned pos = 0;
 
  while( itr + 3 <= buf + length )
    {
      //      if(isTrailer2(itr)) pos = (unsigned)itr; // I hate to brute force this but I had to :-(
      if(isTrailer2(itr)) pos = reinterpret_cast<unsigned long>(itr); 
      ++itr;
    }
  
  return pos; 
}

bool FileReaderSPNewFormat::isTrailer1(const unsigned short* buf)
{
  bool result = false;
  result = (trailer[0] & buf[0]) == trailer[0];
  result = (trailer[1] & buf[1]) == trailer[1];
  result = (trailer[2] & buf[2]) == trailer[2];
  result = (trailer[3] & buf[3]) == trailer[3];
 
  return result;
}

bool FileReaderSPNewFormat::isTrailer2(const unsigned short* buf)
{
  bool result = false;
  result = (trailer[4] & buf[0]) == trailer[4];
  result = (trailer[5] & buf[1]) == trailer[5];
  result = (trailer[6] & buf[2]) == trailer[6];
  result = (trailer[7] & buf[3]) == trailer[7];
  return result;
}

bool FileReaderSPNewFormat::hasTrailer(const unsigned short* buf, unsigned length)
{
  return (isTrailer1(buf + length - 8) && isTrailer2(buf + length - 4));
}
