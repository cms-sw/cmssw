#ifndef DataFormats_L1Scouting_OrbitCollection_h
#define DataFormats_L1Scouting_OrbitCollection_h

#include "DataFormats/Common/interface/traits.h"
#include "FWCore/Utilities/interface/GCCPrerequisite.h"

#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"

#include <cstdint>
#include <vector>

namespace scoutingRun3 {

  template<class T>
  class OrbitCollection {
    public:
      typedef typename std::vector<T>::const_iterator const_iterator;

      // initialize the offset vector with 0s from 0 to 3565.
      // BX range is [1,3564], an extra entry is needed for the offserts of the last BX 
      OrbitCollection(): bxOffsets_(3566, 0), data_(0) {}
      OrbitCollection(
        std::vector<std::vector<T>> &orbitBuffer,
        unsigned nObjects=0): bxOffsets_(3566, 0), data_(nObjects) {
        fillAndClear(orbitBuffer, nObjects);
      }

      OrbitCollection(const OrbitCollection& other) = default;
      OrbitCollection(OrbitCollection&& other) = default;
      OrbitCollection & operator=(const OrbitCollection& other) = default;
      OrbitCollection & operator=(OrbitCollection&& other) = default;

      void swap(OrbitCollection& other){
        using std::swap;
        swap(bxOffsets_, other.bxOffsets_);
        swap(data_, other.data_);
      }

      // Fill the orbit collection starting from a vector of vectors, one per BX.
      // Objects are moved into a flat data vector and a vector is used to keep track
      // of the starting offset of every BX.
      // Input vector must be sorted with increasing BX and contain 3565 elements (BX in [1,3564])
      void fillAndClear(std::vector<std::vector<T>> &orbitBuffer, unsigned nObjects=0){
        if (orbitBuffer.size()!=3565)
          throw cms::Exception("OrbitCollection::fillAndClear") 
            << "Trying to fill the collection by passing an orbit buffer with incorrect size. "
            << "Passed " << orbitBuffer.size() << ", expected 3565" << std::endl;
        data_.reserve(nObjects);
        bxOffsets_[0] = 0;
        unsigned bxIdx = 1;
        for (auto &bxVec: orbitBuffer){
          // increase offset by the currect vec size
          bxOffsets_[bxIdx] = bxOffsets_[bxIdx-1] + bxVec.size();

          // if bxVec contains something, move it into the data_ vector
          // and clear bxVec objects
          if (bxVec.size()>0){
            data_.insert(data_.end(),
              std::make_move_iterator(bxVec.begin()),
              std::make_move_iterator(bxVec.end())
            );
            bxVec.clear();
          }

          // increment bx index
          bxIdx++;
        }
      }

      // iterate over all elements contained in data
      const_iterator begin() const {return data_.begin();}
      const_iterator end() const {return data_.end();}

      // iterate over elements of a bx
      const_iterator begin(unsigned bx) const { return bxRange(bx).first; }
      const_iterator end(unsigned bx) const { return bxRange(bx).second; }

      std::pair<const_iterator, const_iterator> bxRange(unsigned bx) const {
        if (bx>3564)
          throw cms::Exception("OrbitCollection::getBxVectorView") 
            << "Trying to access and object outside the orbit range. "
            << " BX = " << bx << std::endl;

        if (getBxSize(bx)>0){
          return std::make_pair(data_.begin()+bxOffsets_[bx], data_.begin()+bxOffsets_[bx+1]);
        } else {
          return std::make_pair(end(), end());
        }
      }

      // get number of objects stored in a BX
      int getBxSize(unsigned bx) const {
        if (bx>3564){
          edm::LogWarning ("OrbitCollection")
                << "Called getBxSize() of a bx out of the orbit range."
                << " BX = " << bx << std::endl;
          return 0;
        }
        if (data_.size()==0) {
          edm::LogWarning ("OrbitCollection")
              << "Called getBxSize() but collection is empty." << std::endl;
        }
        return bxOffsets_[bx+1] - bxOffsets_[bx];
      }

      // get i-th object from BX
      const T& getBxObject(unsigned bx, unsigned i) const {
        if (bx>3564)
          throw cms::Exception("OrbitCollection::getBxObject") 
            << "Trying to access and object outside the orbit range. "
            << " BX = " << bx << std::endl;
        if (i>=getBxSize(bx))
          throw cms::Exception("OrbitCollection::getBxObject") 
              << "Trying to get element " << i << " but for"
              << " BX = " << bx << " there are " << getBxSize(bx) 
              << " elements." <<std::endl;
        
        return data_[bxOffsets_[bx]+i];
      }

      // get the list of non empty BXs
      std::vector<unsigned> getFilledBxs() const {
        std::vector<unsigned> filledBxVec;
        if (data_.size()>0){
          for (unsigned bx=0; bx<3565; bx++) { 
            if (getBxSize(bx)>0) filledBxVec.push_back(bx);
          }
        }
        return filledBxVec;
      }

      int size() const { return data_.size(); }

    private:
      // store data vector and BX offsets as flat vectors.
      // offset contains one entry per BX, indicating the starting index
      // of the objects for that BX.  
      std::vector<unsigned> bxOffsets_;
      std::vector<T> data_;
  };

  typedef OrbitCollection<ScMuon>    ScMuonOrbitCollection;
  typedef OrbitCollection<ScJet>     ScJetOrbitCollection;
  typedef OrbitCollection<ScEGamma>  ScEGammaOrbitCollection;
  typedef OrbitCollection<ScTau>     ScTauOrbitCollection;
  typedef OrbitCollection<ScEtSum>   ScEtSumOrbitCollection;
}

#endif // DataFormats_L1Scouting_OrbitCollection_h