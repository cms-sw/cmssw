#ifndef FEDRawData_FEDRawDataCollection_h
#define FEDRawData_FEDRawDataCollection_h

/** \class FEDRawDataCollection
 *  An EDCollection storing the raw data for all  FEDs in a Event.
 *  
 *  Reference: DaqPrototype/DaqPersistentData/interface/DaqFEDOpaqueData.h
 *
 *  \author N. Amapane - S. Argiro'
 */

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include "DataFormats/Common/interface/traits.h"
#include "FWCore/Utilities/interface/GCCPrerequisite.h"

#include <vector>

class FEDRawDataCollection : public edm::DoNotRecordParents {
public:
  using container_type = std::vector<FEDRawData>;
  using value_type = container_type::value_type;
  using size_type = container_type::size_type;
  using reference = container_type::reference;
  using const_reference = container_type::const_reference;
  using pointer = container_type::pointer;
  using const_pointer = container_type::const_pointer;
  using iterator = container_type::iterator;
  using const_iterator = container_type::const_iterator;

  FEDRawDataCollection();
  ~FEDRawDataCollection() = default;

  FEDRawDataCollection(const FEDRawDataCollection&) = default;
  FEDRawDataCollection(FEDRawDataCollection&&) = default;
  FEDRawDataCollection& operator=(const FEDRawDataCollection&) = default;
  FEDRawDataCollection& operator=(FEDRawDataCollection&&) = default;

  /// retrieve data for fed @param fedid
  const FEDRawData& FEDData(int fedid) const noexcept { return data_[fedid]; }

  /// retrieve data for fed @param fedid
  FEDRawData& FEDData(int fedid) noexcept { return data_[fedid]; }

  void swap(FEDRawDataCollection& other) { data_.swap(other.data_); }

  auto begin() const noexcept { return data_.begin(); }
  auto end() const noexcept { return data_.end(); }

  auto begin() noexcept { return data_.begin(); }
  auto end() noexcept { return data_.end(); }

  auto cbegin() noexcept { return data_.cbegin(); }
  auto cend() noexcept { return data_.cend(); }

  auto size() noexcept { return data_.size(); }

private:
  container_type data_;  ///< the raw data
};

inline void swap(FEDRawDataCollection& a, FEDRawDataCollection& b) { a.swap(b); }

#endif
