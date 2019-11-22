#include "GeneratorInterface/SherpaInterface/interface/SherpackUtilities.h"
#include <unistd.h>
#include <cstdlib>
namespace spu {

  // functions for inflating (and deflating)

  //~ /* Compress from file source to file dest until EOF on source.
  //~ def() returns Z_OK on success, Z_MEM_ERROR if memory could not be
  //~ allocated for processing, Z_STREAM_ERROR if an invalid compression
  //~ level is supplied, Z_VERSION_ERROR if the version of zlib.h and the
  //~ version of the library linked do not match, or Z_ERRNO if there is
  //~ an error reading or writing the files. */
  int def(FILE *source, FILE *dest, int level) {
    int ret, flush;
    unsigned have;
    z_stream strm;
    unsigned char in[CHUNK];
    unsigned char out[CHUNK];

    /* allocate deflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    ret = deflateInit(&strm, level);
    if (ret != Z_OK)
      return ret;

    /* compress until end of file */
    do {
      strm.avail_in = fread(in, 1, CHUNK, source);
      if (ferror(source)) {
        (void)deflateEnd(&strm);
        return Z_ERRNO;
      }
      flush = feof(source) ? Z_FINISH : Z_NO_FLUSH;
      strm.next_in = in;

      /* run deflate() on input until output buffer not full, finish
           compression if all of source has been read in */
      do {
        strm.avail_out = CHUNK;
        strm.next_out = out;
        ret = deflate(&strm, flush);   /* no bad return value */
        assert(ret != Z_STREAM_ERROR); /* state not clobbered */
        have = CHUNK - strm.avail_out;
        if (fwrite(out, 1, have, dest) != have || ferror(dest)) {
          (void)deflateEnd(&strm);
          return Z_ERRNO;
        }
      } while (strm.avail_out == 0);
      assert(strm.avail_in == 0); /* all input will be used */

      /* done when last data in file processed */
    } while (flush != Z_FINISH);
    assert(ret == Z_STREAM_END); /* stream will be complete */

    /* clean up and return */
    (void)deflateEnd(&strm);
    return Z_OK;
  }

  /* Decompress from file source to file dest until stream ends or EOF.
   inf() returns Z_OK on success, Z_MEM_ERROR if memory could not be
   allocated for processing, Z_DATA_ERROR if the deflate data is
   invalid or incomplete, Z_VERSION_ERROR if the version of zlib.h and
   the version of the library linked do not match, or Z_ERRNO if there
   is an error reading or writing the files. */
  int inf(FILE *source, FILE *dest) {
    int ret;
    unsigned have;
    z_stream strm;
    unsigned char in[CHUNK];
    unsigned char out[CHUNK];

    /* allocate inflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    //~ ret = inflateInit(&strm,15);
    ret = inflateInit2(&strm, (16 + MAX_WBITS));
    if (ret != Z_OK)
      return ret;

    /* decompress until deflate stream ends or end of file */
    do {
      strm.avail_in = fread(in, 1, CHUNK, source);
      if (ferror(source)) {
        (void)inflateEnd(&strm);
        return Z_ERRNO;
      }
      if (strm.avail_in == 0)
        break;
      strm.next_in = in;

      /* run inflate() on input until output buffer not full */
      do {
        strm.avail_out = CHUNK;
        strm.next_out = out;
        ret = inflate(&strm, Z_NO_FLUSH);
        assert(ret != Z_STREAM_ERROR); /* state not clobbered */
        switch (ret) {
          case Z_NEED_DICT:
            ret = Z_DATA_ERROR;
            [[fallthrough]];
          case Z_DATA_ERROR:
          case Z_MEM_ERROR:
            (void)inflateEnd(&strm);
            return ret;
        }
        have = CHUNK - strm.avail_out;
        if (fwrite(out, 1, have, dest) != have || ferror(dest)) {
          (void)inflateEnd(&strm);
          return Z_ERRNO;
        }
      } while (strm.avail_out == 0);

      /* done when inflate() says it's done */
    } while (ret != Z_STREAM_END);

    /* clean up and return */
    (void)inflateEnd(&strm);
    return ret == Z_STREAM_END ? Z_OK : Z_DATA_ERROR;
  }

  /* report a zlib or i/o error */
  void zerr(int ret) {
    fputs("zpipe: ", stderr);
    switch (ret) {
      case Z_ERRNO:
        if (ferror(stdin))
          fputs("error reading stdin\n", stderr);
        if (ferror(stdout))
          fputs("error writing stdout\n", stderr);
        break;
      case Z_STREAM_ERROR:
        fputs("invalid compression level\n", stderr);
        break;
      case Z_DATA_ERROR:
        fputs("invalid or incomplete deflate data\n", stderr);
        break;
      case Z_MEM_ERROR:
        fputs("out of memory\n", stderr);
        break;
      case Z_VERSION_ERROR:
        fputs("zlib version mismatch!\n", stderr);
    }
  }

  /* compress or decompress from stdin to stdout */
  int Unzip(std::string infile, std::string outfile) {
    /////////////////////////////////////////////
    /////////////// BUG FIX FOR MPI /////////////
    /////////////////////////////////////////////
    const char *tmpdir = std::getenv("TMPDIR");
    if (tmpdir && (strlen(tmpdir) > 50)) {
      setenv("TMPDIR", "/tmp", true);
    }
    /////////////////////////////////////////////
    /////////////////////////////////////////////
    /////////////////////////////////////////////
    int ret;
    FILE *in = fopen(infile.c_str(), "r");
    if (!in)
      return -1;
    FILE *out = fopen(outfile.c_str(), "w");
    if (!out)
      return -2;
    /* avoid end-of-line conversions */
    SET_BINARY_MODE(in);
    SET_BINARY_MODE(out);

    ret = inf(in, out);
    if (ret != Z_OK)
      zerr(ret);

    fclose(in);
    fclose(out);
    return ret;
  }

  // functions for untaring Sherpacks
  /* Parse an octal number, ignoring leading and trailing nonsense. */
  int parseoct(const char *p, size_t n) {
    int i = 0;

    while (*p < '0' || *p > '7') {
      ++p;
      --n;
    }
    while (*p >= '0' && *p <= '7' && n > 0) {
      i *= 8;
      i += *p - '0';
      ++p;
      --n;
    }
    return (i);
  }

  /* Returns true if this is 512 zero bytes. */
  int is_end_of_archive(const char *p) {
    int n;
    for (n = 511; n >= 0; --n)
      if (p[n] != '\0')
        return (0);
    return (1);
  }

  /* Create a directory, including parent directories as necessary. */
  void create_dir(char *pathname, int mode) {
    char *p;
    int r;

    /* Strip trailing '/' */
    if (pathname[strlen(pathname) - 1] == '/')
      pathname[strlen(pathname) - 1] = '\0';

    /* Try creating the directory. */
    r = mkdir(pathname, mode);

    if (r != 0) {
      /* On failure, try creating parent directory. */
      p = strrchr(pathname, '/');
      if (p != nullptr) {
        *p = '\0';
        create_dir(pathname, 0755);
        *p = '/';
        r = mkdir(pathname, mode);
      }
    }
    if (r != 0)
      fprintf(stderr, "Could not create directory %s\n", pathname);
  }

  /* Create a file, including parent directory as necessary. */
  FILE *create_file(char *pathname, int mode) {
    FILE *f;
    f = fopen(pathname, "w+");
    if (f == nullptr) {
      /* Try creating parent dir and then creating file. */
      char *p = strrchr(pathname, '/');
      if (p != nullptr) {
        *p = '\0';
        create_dir(pathname, 0755);
        *p = '/';
        f = fopen(pathname, "w+");
      }
    }
    return (f);
  }

  /* Verify the tar checksum. */
  int verify_checksum(const char *p) {
    int n, u = 0;
    for (n = 0; n < 512; ++n) {
      if (n < 148 || n > 155)
        /* Standard tar checksum adds unsigned bytes. */
        u += ((unsigned char *)p)[n];
      else
        u += 0x20;
    }
    return (u == parseoct(p + 148, 8));
  }

  /* Extract a tar archive. */
  void Untar(FILE *a, const char *path) {
    bool longpathname = false;
    bool longlinkname = false;
    char newlongpathname[512];
    char newlonglinkname[512];
    char buff[512];
    FILE *f = nullptr;
    size_t bytes_read;
    int filesize;

    printf("Extracting from %s\n", path);
    for (;;) {
      bytes_read = fread(buff, 1, 512, a);
      if (bytes_read < 512) {
        fprintf(stderr, "Short read on %s: expected 512, got %d\n", path, (int)bytes_read);
        return;
      }
      if (is_end_of_archive(buff)) {
        printf("End of %s\n", path);
        return;
      }
      if (!verify_checksum(buff)) {
        fprintf(stderr, "Checksum failure\n");
        return;
      }
      filesize = parseoct(buff + 124, 12);
      //		printf("%c %d\n",buff[156],filesize);
      switch (buff[156]) {
        case '1':
          printf(" Ignoring hardlink %s\n", buff);
          break;
        case '2':
          if (longpathname && longlinkname) {
            longlinkname = false;
            longpathname = false;
            printf(" Extracting symlink %s\n", newlongpathname);
            symlink(newlonglinkname, newlongpathname);
          } else if (longpathname) {
            longpathname = false;
            printf(" Extracting symlink %s\n", newlongpathname);
            symlink(buff + 157, newlongpathname);
          } else if (longlinkname) {
            longlinkname = false;
            printf(" Extracting symlink %s\n", buff);
            symlink(newlonglinkname, buff);
          } else {
            printf(" Extracting symlink %s\n", buff);
            symlink(buff + 157, buff);
          }
          break;
        case '3':
          printf(" Ignoring character device %s\n", buff);
          break;
        case '4':
          printf(" Ignoring block device %s\n", buff);
          break;
        case '5':
          if (!longpathname) {
            int endposition = -1;
            for (int k = 99; k >= 0; k--) {
              if (buff[k] == '\0')
                endposition = k;
            }
            if (endposition == -1) {
              //~ printf("OLDNAME : %s\n",buff);
              longpathname = true;
              for (int k = 0; k < 100; k++) {
                newlongpathname[k] = buff[k];
              }
              newlongpathname[100] = '\0';
              //~ printf("NEWNAME : %s\n",newlongpathname);
            }
          }

          if (longpathname) {
            printf(" Extracting dir %s\n", newlongpathname);
            create_dir(newlongpathname, parseoct(buff + 100, 8));
            longpathname = false;
          } else {
            printf(" Extracting dir %s\n", buff);
            create_dir(buff, parseoct(buff + 100, 8));
          }
          //~ printf(" Extracting dir %s\n", buff);
          //~ create_dir(buff, parseoct(buff + 100, 8));
          filesize = 0;
          break;
        case '6':
          printf(" Ignoring FIFO %s\n", buff);
          break;
        case 'L':
          longpathname = true;
          //~ printf(" Long Filename found 0 %s\n", buff);
          //~ printf(" Long Filename found 100 %s\n", buff+100);
          //~ printf(" Long Filename found 108 %s\n", buff+108);
          //~ printf(" Long Filename found 116 %s\n", buff+116);
          //~ printf(" Long Filename found 124 %s\n", buff+124);
          //~ printf(" Long Filename found 136 %s\n", buff+136);
          //~ printf(" Long Filename found 148 %s\n", buff+148);
          //~ printf(" Long Filename found 156 %s\n", buff+156);
          //~ printf(" Long Filename found 157 %s\n", buff+157);
          //~ printf(" Long Filename found 158 %s\n", buff+158);
          //~ printf(" Long Filename found 159 %s\n", buff+159);
          //~ printf(" Long Filename found 257 %s\n", buff+257);
          //~ printf(" Long Filename found 263 %s\n", buff+263);
          //~ printf(" Long Filename found 265 %s\n", buff+265);
          //~ printf(" Long Filename found 297 %s\n", buff+297);
          //~ printf(" Long Filename found 329 %s\n", buff+329);
          //~ printf(" Long Filename found 337 %s\n", buff+337);
          //~ printf(" Long Filename found 345 %s\n", buff+345);
          //~ printf(" Long Filename found 346 %s\n", buff+346);
          //~ printf(" Long Filename found 347 %s\n", buff+347);
          break;

        case 'K':
          longlinkname = true;
          break;

        default:
          if (!longpathname) {
            int endposition = -1;
            for (int k = 99; k >= 0; k--) {
              if (buff[k] == '\0')
                endposition = k;
            }
            if (endposition == -1) {
              //~ printf("OLDNAME : %s\n",buff);
              longpathname = true;
              for (int k = 0; k < 100; k++) {
                newlongpathname[k] = buff[k];
              }
              newlongpathname[100] = '\0';
              //~ printf("NEWNAME : %s\n",newlongpathname);
            }
          }
          if (longpathname) {
            printf(" Extracting file %s\n", newlongpathname);
            f = create_file(newlongpathname, parseoct(buff + 100, 8));
            longpathname = false;
          } else {
            printf(" Extracting file %s\n", buff);
            f = create_file(buff, parseoct(buff + 100, 8));
          }
          break;
      }

      if (longlinkname || longpathname) {
        if (buff[156] == 'K') {
          for (int ll = 0; ll < 512; ll++) {
            printf("%c", buff[ll]);
          }
          printf("\n");
          bytes_read = fread(buff, 1, 512, a);
          for (int ll = 0; ll < 512; ll++) {
            printf("%c", buff[ll]);
          }
          printf("\n");
          for (int k = 0; k < filesize; k++) {
            newlonglinkname[k] = buff[k];
          }
          newlonglinkname[filesize] = '\0';
          for (int k = filesize + 1; k < 512; k++) {
            newlonglinkname[k] = '0';
          }
          //~ printf("NEW LinkNAME: %s\n",newlonglinkname);
        } else if (buff[156] == 'L') {
          bytes_read = fread(buff, 1, 512, a);
          for (int k = 0; k < filesize; k++) {
            newlongpathname[k] = buff[k];
          }
          newlongpathname[filesize] = '\0';
          for (int k = filesize + 1; k < 512; k++) {
            newlongpathname[k] = '0';
          }
          //~ printf("NEW FILENAME: %s\n",newlongpathname);
        }
      }

      //~
      //~ if (longpathname) {
      //~ bytes_read = fread(buff, 1, 512, a);
      //~ for (int k=0; k<filesize; k++){
      //~ newlongpathname[k]=buff[k];
      //~ }
      //~ newlongpathname[filesize]='\0';
      //~ for (int k=filesize+1; k<512; k++){
      //~ newlongpathname[k]='0';
      //~ }
      //~ printf("NEW FILENAME: %s\n",newlongpathname);
      //~
      //~ }
      //~ else if (!longpathname && !longlinkname) {
      if (!longpathname && !longlinkname) {
        while (filesize > 0) {
          bytes_read = fread(buff, 1, 512, a);
          if (bytes_read < 512) {
            fprintf(stderr, "Short read on %s: Expected 512, got %d\n", path, (int)bytes_read);
            return;
          }
          if (filesize < 512)
            bytes_read = filesize;
          if (f != nullptr) {
            if (fwrite(buff, 1, bytes_read, f) != bytes_read) {
              fprintf(stderr, "Failed write\n");
              fclose(f);
              f = nullptr;
            }
          }
          filesize -= bytes_read;
        }
        if (f != nullptr) {
          fclose(f);
          f = nullptr;
        }
      }
    }
  }

  // function for calculating the MD5 checksum of a file
  void md5_File(std::string filename, char *result) {
    char buffer[4096];
    MD5_CTX md5;
    MD5_Init(&md5);

    //Open File
    int fd = open(filename.c_str(), O_RDONLY);
    int nb_read;
    while ((nb_read = read(fd, buffer, 4096 - 1))) {
      MD5_Update(&md5, buffer, nb_read);
      memset(buffer, 0, 4096);
    }
    unsigned char tmp[MD5_DIGEST_LENGTH];
    MD5_Final(tmp, &md5);

    //Convert the result
    for (int k = 0; k < MD5_DIGEST_LENGTH; ++k) {
      sprintf(result + k * 2, "%02x", tmp[k]);
    }
  }

}  // End namespace spu
