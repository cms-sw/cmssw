#ifndef FEDRawData_FEDRawDataCollection_h
#define FEDRawData_FEDRawDataCollection_h

/** \class FEDRawDataCollection
 *  An EDCollection storing the raw data for all  FEDs in a Event.
 *  
 *  Reference: DaqPrototype/DaqPersistentData/interface/DaqFEDOpaqueData.h
 *
 *  $Date: 2006/08/20 22:41:11 $
 *  $Revision: 1.6 $
 *  \author N. Amapane - S. Argiro'
 */

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include "DataFormats/Common/interface/traits.h"
#include "FWCore/Utilities/interface/GCCPrerequisite.h"

#include <vector>


class FEDRawDataCollection  {
 public:
  FEDRawDataCollection();

  virtual ~FEDRawDataCollection();
    
  /// retrieve data for fed @param fedid
  const FEDRawData&  FEDData(int fedid) const;

  /// retrieve data for fed @param fedid
  FEDRawData&        FEDData(int fedid);

  FEDRawDataCollection(const FEDRawDataCollection &);

  void swap(FEDRawDataCollection & other) {
    data_.swap(other.data_);
  }

 private:

  std::vector<FEDRawData> data_; ///< the raw data 

};

inline
void swap(FEDRawDataCollection & a, FEDRawDataCollection & b) {
  a.swap(b);
}

#if ! GCC_PREREQUISITE(3,4,4)
namespace edm {
  template <> struct has_swap<FEDRawDataCollection> {
    static bool const value = true;
  };
}
#endif


#endif

