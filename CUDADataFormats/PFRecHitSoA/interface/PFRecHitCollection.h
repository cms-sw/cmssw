#ifndef CUDADataFormats_PFRecHitSoA_interface_PFRecHitCollection_h
#define CUDADataFormats_PFRecHitSoA_interface_PFRecHitCollection_h
//CUDADataFormatsPFRecHitSoA                  PFRecHitSoA   
#include <vector>
#include "CUDADataFormats/CaloCommon/interface/Common.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"

namespace hcal {
  template <typename StoragePolicy>
    struct PFRecHitCollection : public ::calo::common::AddSize<typename StoragePolicy::TagType> {
    PFRecHitCollection() = default;
    PFRecHitCollection(const PFRecHitCollection&) = default;
    PFRecHitCollection& operator=(const PFRecHitCollection&) = default;
    
    PFRecHitCollection(PFRecHitCollection&&) = default;
    PFRecHitCollection& operator=(PFRecHitCollection&&) = default;

    typename StoragePolicy::template StorageSelector<int>::type pfrh_depth;
    typename StoragePolicy::template StorageSelector<int>::type pfrh_layer;
    typename StoragePolicy::template StorageSelector<int>::type pfrh_caloId;
    typename StoragePolicy::template StorageSelector<int>::type pfrh_detId;
    typename StoragePolicy::template StorageSelector<int>::type pfrh_neighbours;
    typename StoragePolicy::template StorageSelector<short>::type pfrh_neighbourInfos;

    typename StoragePolicy::template StorageSelector<float>::type pfrh_time;
    typename StoragePolicy::template StorageSelector<float>::type pfrh_energy;
    typename StoragePolicy::template StorageSelector<float>::type pfrh_pt2;
    typename StoragePolicy::template StorageSelector<float>::type pfrh_x;
    typename StoragePolicy::template StorageSelector<float>::type pfrh_y;
    typename StoragePolicy::template StorageSelector<float>::type pfrh_z;
    
    //m_dEta, m_dPhi, m_repCorners, backPoint

    template <typename U = typename StoragePolicy::TagType>
      typename std::enable_if<std::is_same<U, ::calo::common::tags::Vec>::value, void>::type resize(size_t size) {

      pfrh_depth.resize(size);
      pfrh_layer.resize(size);
      pfrh_caloId.resize(size);
      pfrh_detId.resize(size);
      pfrh_neighbours.resize(8*size);
      
      pfrh_time.resize(size);
      pfrh_energy.resize(size);
      pfrh_pt2.resize(size);
      pfrh_x.resize(size);
      pfrh_y.resize(size);
      pfrh_z.resize(size);
    }
  }; // struct PFRecHitCollection

} // namespace hcal

#endif //CUDADataFormats_HcalPFRecHitSoA_interface_PFRecHitCollection_h
