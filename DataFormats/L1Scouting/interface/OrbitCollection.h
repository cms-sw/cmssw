#ifndef DataFormats_L1Scouting_OrbitCollection_h
#define DataFormats_L1Scouting_OrbitCollection_h

#include "DataFormats/Common/interface/traits.h"
#include "FWCore/Utilities/interface/GCCPrerequisite.h"

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"

#include <cstdint>
#include <vector>

namespace scoutingRun3 {

  template<class T>
  class OrbitCollection {
    public:
      OrbitCollection(): bxOffsets_(3565, 0), bxData_(3565), nObjects_(0) {}

      // append one object to vector at bx
      void addBxObject(int bx, T& object) {
        assert(bx<=3564);
        bxData_[bx].emplace_back(object);
        nObjects_ ++;
      }

      // append objects to bx from an iterator
      template <typename VI>
      void addBxObjects(int bx, VI objectsBegin, VI objectsEnd){
        assert(bx<=3564);
        bxData_[bx].insert(bxData_[bx].end(), objectsBegin, objectsEnd);
        nObjects_ += std::distance(objectsBegin, objectsEnd);
      }      

      // flatten bxData_ vector. Must be called at the end of the orbit.
      void flatten(){
        data_.reserve(nObjects_);
        bxOffsets_[0] = 0;
        int bxIdx = 1;
        for (auto &bxVec: bxData_){
          data_.insert(data_.end(),
            std::make_move_iterator(bxVec.begin()),
            std::make_move_iterator(bxVec.end())
          );
          // increase offset by the currect vec size
          bxOffsets_[bxIdx] = bxOffsets_[bxIdx-1] + bxVec.size();
          bxIdx++;
        }

        bxData_.clear();
      }

      // get number of objects stored in a BX
      int getBxSize(int bx){
        if (bx>3564){
          edm::LogWarning ("OrbitCollection")
                << "Called getBxSize() of a bx out of the orbit range."
                << " BX = " << bx;
          return 0;
        }
        if (data_.size()==0) {
          edm::LogWarning ("OrbitCollection")
              << "Called getBxSize() but data_ is empty.";
        }
        return bxOffsets_[bx+1] - bxOffsets_[bx];
      }

      // get i-th object from BX
      const T& getBxObject(int bx, unsigned i){
        assert(bx<=3564);
        assert(i<getBxSize(bx));
        return data_[bxOffsets_[bx]+i];
      }

    private:
      // store data vector and BX offsets as flat vectors.
      // offset contains one entry per BX, indicating the starting index
      // of the objects for that BX.  
      std::vector<int> bxOffsets_;
      std::vector<T> data_;

      // Transient container used while filling the orbit collection.
      // Needed because data could be added to the collection with unsorted BX.
      // This will not be persisted (transient="true")
      mutable std::vector<std::vector<T>> bxData_;

      // count number of objects inserted into the collection
      int nObjects_;
  };

  typedef OrbitCollection<l1t::Muon>    MuonOrbitCollection;
  typedef OrbitCollection<l1t::Jet>     JetOrbitCollection;
  typedef OrbitCollection<l1t::EGamma>  EGammaOrbitCollection;
  typedef OrbitCollection<l1t::Tau>     TauOrbitCollection;
  typedef OrbitCollection<l1t::EtSum>   EtSumOrbitCollection;

  typedef OrbitCollection<scoutingRun3::ScMuon>        ScMuonOrbitCollection;
  typedef OrbitCollection<scoutingRun3::ScJet>  ScJetOrbitCollection;
  typedef OrbitCollection<scoutingRun3::ScEGamma>  ScEGammaOrbitCollection;
  typedef OrbitCollection<scoutingRun3::ScTau>  ScTauOrbitCollection;
  typedef OrbitCollection<scoutingRun3::ScEtSum>       ScEtSumOrbitCollection;
}

#endif // DataFormats_L1Scouting_OrbitCollection_h