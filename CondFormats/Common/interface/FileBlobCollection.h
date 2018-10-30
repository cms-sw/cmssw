#ifndef Alignment_MillePedeAlignmentAlgorithm_FileBlobCollection_h
#define Alignment_MillePedeAlignmentAlgorithm_FileBlobCollection_h

// Original Author:  Broen van Besien
//         Created:  Mon, 06 Jul 2015 12:18:35 GMT

/*
 * This class...
 * BVB: TODO
 *
 */

#include "CondFormats/Common/interface/FileBlob.h"

#include <vector>

class FileBlobCollection {
 public:
  FileBlobCollection() {};
  ~FileBlobCollection() {};
  void addFileBlob(FileBlob &fileBlob);
  int size() const;
  std::vector<FileBlob>::const_iterator begin() const;
  std::vector<FileBlob>::const_iterator end() const;
  bool mergeProduct(FileBlobCollection const &other);
  void swap(FileBlobCollection& iOther);

 private:
  std::vector<FileBlob> fileBlobs;
};

#endif
