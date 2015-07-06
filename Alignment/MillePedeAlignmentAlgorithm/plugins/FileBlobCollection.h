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
  FileBlobCollection();
  // TODO: BVB probably nicer to have a more "direct" constructor here
  ~FileBlobCollection() {};
  bool mergeProduct(FileBlobCollection const &other);

 private:
  std::vector<FileBlob> fileBlobs;
};

#endif
