#ifndef SherpackUtilities_h
#define SherpackUtilities_h

/* Based on the zlib example code zpipe.c
 * Modified for the use of unzipping Sherpacks
 * Sebastian Thüer, Markus Merschmeyer
 * III. Phys. Inst. A, RWTH Aachen University
 * version 1.0, 1st August 2012
*/

#include <cstdlib>
#include <cstdio>
#include <string>
#include <cstring>
#include <cassert>
#include <zlib.h>
/* This is for mkdir(); this may need to be changed for some platforms. */
#include <sys/stat.h> /* For mkdir() */
#include <openssl/md5.h>
#include <fcntl.h>

#define SET_BINARY_MODE(file)

#define CHUNK 16384

namespace spu {

  // functions for inflating (and deflating) -> (un)zipping the Sherpacks (taken from zlib and zpipe.c)
  int def(FILE*, FILE*, int);
  int inf(FILE*, FILE*);
  void zerr(int);
  int Unzip(std::string, std::string);

  // functions for untaring Sherpacks, based on
  // http://www.opensource.apple.com/source/libarchive/libarchive-23/libarchive/contrib/untar.c
  // ...but heavily modified for long path names...
  /* Parse an octal number, ignoring leading and trailing nonsense. */
  int parseoct(const char*, size_t);
  /* Returns true if this is 512 zero bytes. */
  int is_end_of_archive(const char*);
  /* Create a directory, including parent directories as necessary. */
  void create_dir(char*, int);
  /* Create a file, including parent directory as necessary. */
  FILE* create_file(char*, int);
  /* Verify the tar checksum. */
  int verify_checksum(const char*);
  /* Extract a tar archive. */
  void Untar(FILE*, const char*);

  // function for calculating the MD5 checksum of a file
  void md5_File(std::string, char*);

}  // End namespace spu
#endif
