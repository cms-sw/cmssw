#include "CondFormats/Common/interface/FileBlobCollection.h"

void FileBlobCollection::addFileBlob(FileBlob &fileBlob) {
  fileBlobs.push_back(fileBlob);
}

bool FileBlobCollection::mergeProduct(FileBlobCollection const &other) {
  fileBlobs.insert(fileBlobs.end(), other.fileBlobs.begin(),
                   other.fileBlobs.end());
  return true;
}
