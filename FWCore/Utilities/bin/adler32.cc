/*


  Program to calculate adler32 checksum for every file given
  on the command line. Uses zlib's adler32 routine.

  For the CMS Experiment http://cms.cern.ch

  Author: Stephen J. Gowdy <gowdy@cern.ch>
  Created: 6th Dec 2008

*/

#define _LARGEFILE64_SOURCE
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fmt/format.h>
#include <iostream>
#include <libgen.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>
#ifdef __APPLE__
typedef off_t off64_t;
#define O_LARGEFILE 0
#endif

constexpr int EDMFILEUTILADLERBUFSIZE = 10 * 1024 * 1024;  // 10MB buffer

int main(int argc, char* argv[]) {
  if (argc == 1) {
    std::cout << basename(argv[0]) << ": no files specified.\n";
    exit(1);
  }

  std::unique_ptr<unsigned char[]> buffer{new unsigned char[EDMFILEUTILADLERBUFSIZE]};
  int fileNum = 0;
  for (fileNum = 1; fileNum < argc; fileNum++) {
    uLong adlerCksum = adler32(0, nullptr, 0);
    off64_t fileSize = 0;

    int myFD = open(argv[fileNum], O_RDONLY | O_LARGEFILE);
    if (myFD == -1) {
      std::cout << basename(argv[0]) << ": failed to open file " << argv[fileNum] << ".\n";
      continue;
    }

    lseek(myFD, 0, SEEK_SET);

    int readSize = 0;
    while ((readSize = read(myFD, buffer.get(), EDMFILEUTILADLERBUFSIZE)) > 0) {
      adlerCksum = adler32(adlerCksum, buffer.get(), readSize);
      fileSize += readSize;
    }

    std::cout << fmt::format("{:x} {} {}\n", adlerCksum, fileSize, argv[fileNum]);
  }

  return (0);
}

/* Compile-line: gcc adler32.c -o adler32 -lz */
