#ifndef GENERS_STRINGARCHIVEIO_HH_
#define GENERS_STRINGARCHIVEIO_HH_

#include "Alignment/Geners/interface/StringArchive.hh"

namespace gs {
    // The following function returns "true" on success,
    // "false" on failure.
    bool writeStringArchive(const StringArchive& ar, const char* filename);

    // The following function either succeeds or throws
    // an exception. The archive is created on the heap
    // and eventually has to be deleted by the user.
    StringArchive* readStringArchive(const char* filename);

    // Similar operations with compression
    bool writeCompressedStringArchive(
        const StringArchive& ar, const char* filename,
        unsigned compressionMode = 1U, int compressionLevel = -1,
        unsigned minSizeToCompress = 1024U, unsigned bufSize = 1048576U);

    StringArchive* readCompressedStringArchive(const char* filename);

    // The following function will write a compressed string archive
    // using default compression parameters if the file name has the
    // given suffix, otherwise it will write an uncompressed archive.
    // If the suffix is not provided (i.e., default value of 0 is used),
    // ".gssaz" will be assumed.
    bool writeCompressedStringArchiveExt(const StringArchive& ar,
                                         const char* filename,
                                         const char* suffix = 0);

    // The following function will attempt to read a compressed string
    // archive if the file name has the given suffix, otherwise it will
    // attempt to read an uncompressed archive. If the suffix is not
    // provided (i.e., default value of 0 is used), ".gssaz" will be assumed.
    StringArchive* readCompressedStringArchiveExt(const char* filename,
                                                  const char* suffix = 0);

    // This function will extract one string archive from another
    StringArchive* loadStringArchiveFromArchive(AbsArchive& arch,
                                                unsigned long long id);
}

#endif // GENERS_STRINGARCHIVEIO_HH_

