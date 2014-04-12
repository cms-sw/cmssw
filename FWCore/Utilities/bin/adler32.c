/*


  Program to calculate adler32 checksum for every file given
  on the command line. Uses zlib's adler32 routine.

  For the CMS Experiment http://cms.cern.ch

  Author: Stephen J. Gowdy <gowdy@cern.ch>
  Created: 6th Dec 2008

*/

#define _LARGEFILE64_SOURCE
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <zlib.h>
#include <libgen.h>
#ifdef __APPLE__
typedef off_t off64_t;
#define O_LARGEFILE 0
#endif

#define EDMFILEUTILADLERBUFSIZE 10*1024*1024 // 10MB buffer

int main(int argc, char* argv[])
{
  unsigned char* buffer = malloc( EDMFILEUTILADLERBUFSIZE );

  if ( argc == 1 )
    {
      printf( "%s: no files specified.\n", basename( argv[0] ) );
      free( buffer );
      exit(1);
    }

  int fileNum = 0;
  for ( fileNum = 1; fileNum < argc; fileNum++ )
    {
      uLong adlerCksum = adler32( 0, 0, 0 );
      off64_t fileSize = 0;

      int myFD = open( argv[fileNum], O_RDONLY | O_LARGEFILE );
      if ( myFD == -1 )
	{
	  printf( "%s: failed to open file %s.\n", basename( argv[0] ),
		  argv[fileNum] );
	  continue;
	}

      lseek( myFD, 0, SEEK_SET );

      int readSize = 0;
      while ( ( readSize = read( myFD, buffer, EDMFILEUTILADLERBUFSIZE ) )
	      > 0 )
	{
	  adlerCksum = adler32( adlerCksum, buffer, readSize );
	  fileSize += readSize;
	}

      /* This is rather ugly but seems needed */
#if __WORDSIZE == 64
      printf ( "%lx %ld %s\n", adlerCksum, fileSize, argv[fileNum] );
#else
      printf ( "%lx %lld %s\n", adlerCksum, fileSize, argv[fileNum] );
#endif
    }

  free( buffer );
  return(0);
}

/* Compile-line: gcc adler32.c -o adler32 -lz */
