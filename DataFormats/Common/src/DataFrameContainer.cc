#include "DataFormats/Common/interface/DataFrameContainer.h"

#include <boost/iterator/permutation_iterator.hpp>
#include <boost/bind.hpp>

#include <algorithm>
#include <numeric>
#include <cstdlib>


namespace edm {

  void DataFrameContainer::sort() {
    if (size()<2) return;
    std::vector<int> indices(size(),1);
    indices[0]=0;
    std::partial_sum(indices.begin(),indices.end(),indices.begin());
    std::sort(indices.begin(), indices.end(),
	      boost::bind(std::less<id_type>(),
			  boost::bind<id_type const &>(&IdContainer::operator[],boost::ref(m_ids),_1),
			  boost::bind<id_type const &>(&IdContainer::operator[],boost::ref(m_ids),_2)
			  )
	      );
    
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
