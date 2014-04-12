#ifndef DataFormats_Common_OneToManyWithQuality_h
#define DataFormats_Common_OneToManyWithQuality_h
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"

namespace edm {
  template<typename CKey, typename CVal, typename Q, typename index = unsigned int>
  class OneToManyWithQuality : public OneToManyWithQualityGeneric<CKey, CVal, Q, index> {
  private:
    typedef OneToManyWithQualityGeneric<CKey, CVal, Q, index> base;
 
  public:
    /// values reference collection type
    typedef typename base::val_type val_type;
    /// insert key type
    typedef typename base::key_type key_type;
    /// insert val type
    typedef typename base::data_type data_type;
    /// index type
    typedef typename base::index_type index_type;
    /// map type
    typedef typename base::map_type map_type;
    /// reference set type
    typedef typename base::ref_type ref_type;
    /// transient map type
    typedef typename base::transient_map_type transient_map_type;
    /// transient key vector
    typedef typename base::transient_key_vector transient_key_vector;
    /// transient val vector
    typedef typename base::transient_val_vector transient_val_vector;  
  };
}

#endif
