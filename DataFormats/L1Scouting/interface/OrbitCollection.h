#ifndef DataFormats_L1Scouting_OrbitCollection_h
#define DataFormats_L1Scouting_OrbitCollection_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Span.h"

#include <cstdint>
#include <vector>

template <class T>
class OrbitCollection {
public:
  typedef typename std::vector<T>::iterator iterator;
  typedef typename std::vector<T>::const_iterator const_iterator;
  typedef T value_type;
  typedef typename std::vector<T>::size_type size_type;

  // Initialize the offset vector with 0s from 0 to 3565.
  // BX range is [1,3564], an extra entry is needed for the offserts of the last BX
  OrbitCollection() : bxOffsets_(orbitBufferSize_ + 1, 0), data_(0) {}
  // Construct the flat orbit collection starting from an OrbitBuffer.
  // The method fillAndClear will be used, meaning that, after copying the objects,
  // orbitBuffer's vectors will be cleared.
  OrbitCollection(std::vector<std::vector<T>>& orbitBuffer, unsigned nObjects = 0)
      : bxOffsets_(orbitBufferSize_ + 1, 0), data_(nObjects) {
    fillAndClear(orbitBuffer, nObjects);
  }

  OrbitCollection(const OrbitCollection& other) = default;
  OrbitCollection(OrbitCollection&& other) = default;
  OrbitCollection& operator=(const OrbitCollection& other) = default;
  OrbitCollection& operator=(OrbitCollection&& other) = default;

  // Fill the orbit collection starting from a vector of vectors, one per BX.
  // Objects are copied into a flat data vector, and a second vector is used to keep track
  // of the starting index in the data vector for every BX.
  // After the copy, the original input buffer is cleared.
  // Input vector must be sorted with increasing BX and contain 3565 elements (BX in [1,3564])
  void fillAndClear(std::vector<std::vector<T>>& orbitBuffer, unsigned nObjects = 0) {
    if (orbitBuffer.size() != orbitBufferSize_)
      throw cms::Exception("OrbitCollection::fillAndClear")
          << "Trying to fill the collection by passing an orbit buffer with incorrect size. "
          << "Passed " << orbitBuffer.size() << ", expected 3565";
    data_.reserve(nObjects);
    bxOffsets_[0] = 0;
    unsigned bxIdx = 1;
    for (auto& bxVec : orbitBuffer) {
      // increase offset by the currect vec size
      bxOffsets_[bxIdx] = bxOffsets_[bxIdx - 1] + bxVec.size();

      // if bxVec contains something, copy it into the data_ vector
      // and clear original bxVec objects
      if (bxVec.size() > 0) {
        data_.insert(data_.end(), bxVec.begin(), bxVec.end());
        bxVec.clear();
      }

      // increment bx index
      bxIdx++;
    }
  }

  // iterate over all elements contained in data
  const_iterator begin() const { return data_.begin(); }
  const_iterator end() const { return data_.end(); }

  // iterate over elements of a bx
  edm::Span<const_iterator> bxIterator(unsigned bx) const {
    if (bx >= orbitBufferSize_)
      throw cms::Exception("OrbitCollection::bxIterator") << "Trying to access and object outside the orbit range. "
                                                          << " BX = " << bx;
    if (getBxSize(bx) > 0) {
      return edm::Span(data_.begin() + bxOffsets_[bx], data_.begin() + bxOffsets_[bx + 1]);
    } else {
      return edm::Span(end(), end());
    }
  }

  // get number of objects stored in a BX
  int getBxSize(unsigned bx) const {
    if (bx >= orbitBufferSize_) {
      cms::Exception("OrbitCollection") << "Called getBxSize() of a bx out of the orbit range."
                                        << " BX = " << bx;
      return 0;
    }
    return bxOffsets_[bx + 1] - bxOffsets_[bx];
  }

  // get i-th object from BX
  const T& getBxObject(unsigned bx, unsigned i) const {
    if (bx >= orbitBufferSize_)
      throw cms::Exception("OrbitCollection::getBxObject") << "Trying to access and object outside the orbit range. "
                                                           << " BX = " << bx;
    if (i >= getBxSize(bx))
      throw cms::Exception("OrbitCollection::getBxObject")
          << "Trying to get element " << i << " but for"
          << " BX = " << bx << " there are " << getBxSize(bx) << " elements.";

    return data_[bxOffsets_[bx] + i];
  }

  // get the list of non empty BXs
  std::vector<unsigned> getFilledBxs() const {
    std::vector<unsigned> filledBxVec;
    if (!data_.empty()) {
      for (unsigned bx = 0; bx < orbitBufferSize_; bx++) {
        if ((bxOffsets_[bx + 1] - bxOffsets_[bx]) > 0)
          filledBxVec.push_back(bx);
      }
    }
    return filledBxVec;
  }

  int size() const { return data_.size(); }

  T& operator[](std::size_t i) { return data_[i]; }
  const T& operator[](std::size_t i) const { return data_[i]; }

  // used by ROOT storage
  CMS_CLASS_VERSION(3)

private:
  // store data vector and BX offsets as flat vectors.
  // offset contains one entry per BX, indicating the starting index
  // of the objects for that BX.
  std::vector<unsigned> bxOffsets_;
  std::vector<T> data_;

  // there are 3564 BX in one orbtit [1,3564], one extra
  // count added to keep first entry of the vector
  static constexpr int orbitBufferSize_ = 3565;
};

#endif  // DataFormats_L1Scouting_OrbitCollection_h
