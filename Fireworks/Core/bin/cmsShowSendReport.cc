#include <stdio.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include "zlib.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#include <string.h>
#include <netdb.h>
#include <iostream>

#define BUFLEN 60000

// AMT: This code is a substitute of netcat command. The reason it is replaced 
//      is that netcat has limited buffer to 1024 bytes.
//      
//      TODO:: waith for server echo with timeout.


void getCompressedBuffer(const char* fname, Bytef** buffPtr, unsigned long& zippedSize)
{
   FILE* pFile = fopen ( fname , "r" );
   if ( pFile==NULL )  { std::cerr << "Can't open " << fname << std::endl; exit(1); }

   // obtain file size:
   fseek (pFile , 0 , SEEK_END);
   unsigned int lSize = ftell (pFile);
   rewind (pFile);

   // allocate memory to contain the whole file:
   void* buffer = malloc (sizeof(Bytef)*(lSize));

   size_t result = fread (buffer, 1, lSize ,pFile);
   fclose(pFile);
   if ( !result ) { std::cerr << "Failed to read " << fname <<std::endl; exit(1); }

   //
   // write a new buffer. First four bytes is integer with  
   // value of size of uncompressed data. Remaining content 
   // is compressed original buffer.
   //
   unsigned int deflatedSize =  compressBound(lSize) + 4; // estimation
   Bytef * deflatedBuff = (Bytef*) malloc (sizeof(Bytef)*(deflatedSize));
   *((unsigned int*)deflatedBuff) = htonl(lSize);

   //set buffer ptr
   *buffPtr = deflatedBuff;

   // compress buffer
   zippedSize = deflatedSize;
   compress(deflatedBuff+4, &zippedSize, (const Bytef *)buffer, lSize);
   zippedSize +=4;

   /*
   printf("zipped size %d \n", (int)zippedSize);
   FILE* pFileOut = fopen ( "myfile-compressed" , "wb" );
   fwrite (deflatedBuff , 1 , zippedSize , pFileOut );
   fclose(pFileOut);
   */
}

int main(int argc, char **argv)
{
  if (argc != 2)
  {
      std::cerr << "Uasage: sendCrashReport <fileName>" << std::endl; exit(1);
  }

   // socket creation
   int sd = socket(AF_INET,SOCK_DGRAM, 0);
   if (sd  < 0) { return 1; }

   // printf("bind port\n");
   struct sockaddr_in cliAddr;
   cliAddr.sin_family = AF_INET;
   cliAddr.sin_addr.s_addr = htonl(INADDR_ANY);
   cliAddr.sin_port = htons(0);

   int rc = bind(sd, (struct sockaddr *) &cliAddr, sizeof(cliAddr)); 
   if (rc < 0) {
      std::cerr << "Can't bind port %d " << rc << std::endl; exit(1);
   }
   
   // send data 
   struct hostent* h = gethostbyname("xrootd.t2.ucsd.edu");
   if (!h) {
      std::cerr << "Can't get gost ip \n"; exit(1);
   }

   struct sockaddr_in remoteServAddr;
   remoteServAddr.sin_family = h->h_addrtype;
   memcpy((char *) &remoteServAddr.sin_addr.s_addr, h->h_addr_list[0], h->h_length);
   remoteServAddr.sin_port = htons(9699);

   Bytef* buff;
   unsigned long  buffSize;
   getCompressedBuffer(argv[1], &buff, buffSize);

   int res = sendto(sd, buff, buffSize, 0, 
                    (struct sockaddr *) &remoteServAddr, 
                    sizeof(remoteServAddr));
   delete buff;
   
   if (res == -1)
      std::cerr << "Sending report has failed." << std::endl;
   else
      std::cout << "Report has been sent." <<std::endl;

}
