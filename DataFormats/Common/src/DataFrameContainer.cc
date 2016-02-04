#include "DataFormats/Common/interface/DataFrameContainer.h"

#include <boost/iterator/permutation_iterator.hpp>

#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <cstring>

namespace edm {
  namespace {
    struct TypeCompare {
       typedef DataFrameContainer::id_type id_type;
       std::vector<id_type> const& ids_;
       TypeCompare(std::vector<id_type> const& iType): ids_(iType) {}
        bool operator()(id_type const& iLHS, id_type const& iRHS) const {
          return ids_[iLHS] < ids_[iRHS];
        }
    };
  }

  void DataFrameContainer::sort() {
    if (size()<2) return;
    std::vector<int> indices(size(),1);
    indices[0]=0;
    std::partial_sum(indices.begin(),indices.end(),indices.begin());
    std::sort(indices.begin(), indices.end(), TypeCompare(m_ids));
    {
      IdContainer tmp(m_ids.size());
      std::copy(
		boost::make_permutation_iterator( m_ids.begin(), indices.begin() ),
		boost::make_permutation_iterator( m_ids.end(), indices.end() ),
		tmp.begin());
      tmp.swap(m_ids);
    }
    {
      //      std::transform(indices.begin(),indices.end(),indices.begin(),
      //	     boost::bind(std::multiplies<int>(),m_stride,_1));
      DataContainer tmp(m_data.size());
      size_type s = m_stride*sizeof(data_type);
      for(size_type j=0, i=0; i!=indices.size(); ++i, j+=m_stride)
	::memcpy(&tmp[j], &m_data[indices[i]*m_stride], s);
      tmp.swap(m_data);
    }
    
  }
  
}
