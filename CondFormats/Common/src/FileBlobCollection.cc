#include "CondFormats/Common/interface/FileBlobCollection.h"

void FileBlobCollection::addFileBlob(FileBlob &fileBlob) {
  fileBlobs.push_back(fileBlob);
}

bool FileBlobCollection::mergeProduct(FileBlobCollection const &other) {
  std::cout << "+++++ Length of fileBlobs is: " << fileBlobs.size()
            << std::endl;
  // fileBlobs.push_back(other.fileBlobs.front());
  std::cout << "+++++ Using multi method!" << std::endl;
  fileBlobs.insert(fileBlobs.end(), other.fileBlobs.begin(),
                   other.fileBlobs.end());
  std::cout << "+++++ Length of fileBlobs is: " << fileBlobs.size()
            << std::endl;
  // TODO: Should also work for more than one
  return true;
}
