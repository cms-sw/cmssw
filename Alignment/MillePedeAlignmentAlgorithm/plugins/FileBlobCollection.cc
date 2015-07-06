#include "Alignment/MillePedeAlignmentAlgorithm/plugins/FileBlobCollection.h"

FileBlobCollection::FileBlobCollection(){
  std::cout << "Constructor";
  std::cout << "Length of fileBlobs is: " << fileBlobs.size();
}

bool FileBlobCollection::mergeProduct(FileBlobCollection const &other) {
  std::cout << "Length of fileBlobs is: " << fileBlobs.size();
  fileBlobs.push_back(other.fileBlobs.front());
  std::cout << "Length of fileBlobs is: " << fileBlobs.size();
  // TODO: Should work for more than one
  return true;
}
