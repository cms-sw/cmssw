#include <IORawData/CSCTFCommissioning/src/FileReaderSP.h>

#include <iostream>
#include <fstream>
#include <cstdio>
#include <unistd.h>
#include <cstdlib>
#include <cstring>

FileReaderSP::FileReaderSP()
{
  pointer = 0;
  key[0] = 0xf000;
  key[1] = 0x7fff;
  key[2] = 0x7fff;
}

void FileReaderSP::Configure() {
}


int FileReaderSP::chunkSize() {
  /*int cnt[2];
  read(fd_schar,cnt,4);
  return cnt[0];*/
  return 0;
}


int FileReaderSP::readSP(unsigned short **buf, const bool debug) {
 
  int counter=0;
  unsigned short tmp[1];
  unsigned short holder =0;
  unsigned short a[6000];
  int bytes_read=0;

  do{
    if(pointer == 0)
      {
	bytes_read = read(fd_schar,tmp,2);  //read only 1 line == 2 bytes of data
	if(bytes_read != 2)
	  {
	    cout<<"FileReaderSP :: Reached end of file!!!\n";
	    return 0;
	  }
	a[counter]=tmp[0];
      }
    else
      {
	holder = miniBuf.front();
	a[counter] = holder;
	miniBuf.pop();
	holder = 0;
      }

    counter++;
    if((bytes_read = fillMiniBuf())== -1)
      {
	while(miniBuf.size()>0)
	  {
	    holder = miniBuf.front();
	    a[counter] = holder;
	    miniBuf.pop();
	    holder = 0;
	    counter++;    
	  }
	void * last = malloc((counter-1)*sizeof(unsigned short));
	memset(last,0,(counter-1)*sizeof(unsigned short));
	memcpy(last,a,(counter-1)*sizeof(unsigned short));
	*buf=reinterpret_cast<unsigned short*>(last);
	pointer = 0;
	return (counter-1)*sizeof(unsigned short);
      }
    pointer += bytes_read;

  }while(!isHeader());

  void * event = malloc(counter*sizeof(unsigned short));
  memset(event,0,counter*sizeof(unsigned short));
  memcpy(event,a,counter*sizeof(unsigned short));
  *buf=reinterpret_cast<unsigned short*>(event);

  
  
  //printf("\n\n\n  returning %d: \n\n\n",(count*8));

  return counter*sizeof(unsigned short);  //Total number of bytes read

} //readSP

int FileReaderSP::fillMiniBuf()
{
  int count=0;
  unsigned short temp[1];
  while(miniBuf.size() < 6)
    {
      if(read(fd_schar,temp,2)==2)
	{
	  miniBuf.push(temp[0]);
	  count++;
	}
      else
	{
	  return -1;
	}
    }
  return count;
}

bool FileReaderSP::isHeader()
{
  int holder = 0;
  for(int i=0;i<MAXBUFSIZE;i++)
    {
      holder = miniBuf.front();
      refBuf[i] = holder;
      miniBuf.pop();
      miniBuf.push(refBuf[i]);
    }    

  return ((refBuf[0]&key[0])==key[0] && (refBuf[1]&key[0])==key[0] && (refBuf[2]&key[0])==key[0]
	  && (refBuf[3]&key[0])==key[0] && (refBuf[4]|key[1])==key[1] && (refBuf[5]|key[2])==key[2]);
}
