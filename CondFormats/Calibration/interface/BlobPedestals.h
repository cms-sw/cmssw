#ifndef BLOBPEDESTALS_H
#define BLOBPEDESTALS_H
#include<vector>
class BlobPedestals {
public:
  BlobPedestals();
  virtual ~BlobPedestals();
  std::vector<unsigned int>  m_pedestals;
};
#endif
