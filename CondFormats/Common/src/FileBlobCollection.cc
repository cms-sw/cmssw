#include "CondFormats/Common/interface/FileBlobCollection.h"

void FileBlobCollection::addFileBlob(FileBlob &fileBlob) {
  fileBlobs.push_back(fileBlob);
}

int FileBlobCollection::size() const {
  return fileBlobs.size();
}

std::vector<FileBlob>::const_iterator FileBlobCollection::begin() const {
  return fileBlobs.begin();
}

std::vector<FileBlob>::const_iterator FileBlobCollection::end() const {
  return fileBlobs.end();
}

bool FileBlobCollection::mergeProduct(FileBlobCollection const &other) {
  fileBlobs.insert(fileBlobs.end(), other.fileBlobs.begin(),
                   other.fileBlobs.end());
  return true;
}
