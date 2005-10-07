#ifndef DataFormats_MuonDigiCollection_h
#define DataFormats_MuonDigiCollection_h

/**
 * \file
 * Declaration of class MuonDigiCollection
 *
 * \author Stefano ARGIRO
 * \version $Id: MuonDigiCollection.h,v 1.1 2005/08/23 09:09:06 argiro Exp $
 * \date 05 Aug 2005
 */


#include <vector>
#include <map>
#include <boost/iterator/transform_iterator.hpp>

/**
 * \class DigiContainerIteratorAdaptor MuonDigiCollection.h "/MuonDigiCollection.h"
 * 
 * \brief An iterator adaptor for a map<Index, vector<Digi> >
 * that when dereferenced returns a 
 * pair<Index, pair<vector<Digi>::const_iterator,
 * vector<Digi>::const_iterator > > 
 * where the two iterators point to begin and and of the vector
 *
 * \author Stefano ARGIRO
 * \date 05 Aug 2005
*/

template <typename IndexType, 
	  typename DigiType>
  class DigiContainerIteratorAdaptor {
 
  public:

    typedef typename std::vector<DigiType>::const_iterator const_iterator;
    typedef typename std::pair<const_iterator,const_iterator> Range;
    typedef typename std::map<IndexType,std::vector<DigiType> > BaseContainer;
    typedef typename BaseContainer::const_iterator BaseIterator;

    class Dereference {
      
    public:

      typedef typename std::pair<IndexType,Range> result_type;
  
      result_type operator()(typename BaseIterator::reference thePair) const { 
	
	return std::make_pair(thePair.first,
			      std::make_pair(thePair.second.begin(), 
					     thePair.second.end()) 
			      ); }
    };// class Dereference 
    
    typedef  typename 
    ::boost::transform_iterator<Dereference,
				BaseIterator,
				typename Dereference::result_type > type;
   
    /// takes the original iterator and returns the adapted one
    static type adapt(BaseIterator iter){
      return type(iter);
    }

  }; // DigiContainerIteratorAdaptor



/**
 * \class MuonDigiCollection MuonDigiCollection.h "/MuonDigiCollection.h"
 *
 * \brief A container for a generic type of digis indexed by 
 *          some index, implemented with a map<IndexType, vector<DigiType> > 
 *
 *   Example: 
 *
 *   \code
 *   typedef MuonDigiCollection<DTDetId,DTDigi> DTDigiCollection
 *   \endcode
 *
 *   \note: Requirements
 *   - IndexType must provide operator <    
 *
 *   \author Stefano ARGIRO
 *   \date 05 Aug 2005
 */

template <typename IndexType, 
	  typename DigiType>

class MuonDigiCollection {
  
public:

  MuonDigiCollection(){}

  typedef typename std::vector<DigiType>::const_iterator const_iterator;
  typedef typename std::pair<const_iterator,const_iterator> Range;
  
  
  /// insert a digi for a given DetUnit  @deprecated 
  void insertDigi(const IndexType& index, const DigiType& digi){
  }
  
  /// insert a range of digis for a  given DetUnit
  void put(Range range, const IndexType& index){
    std::vector<DigiType>& digis = data_[index];
    size_t size = digis.size();
    digis.resize(size + (range.second - range.first));
    std::copy(range.first, range.second,digis.begin());
  }
 
  /// return the digis for a given DetUnit 
  Range get(const IndexType& index) const{
 
    typename container::const_iterator it = data_.find(index);
    const std::vector<DigiType>& digis = (*it).second;
    return std::make_pair(digis.begin(),digis.end());
  }
  
  typedef typename DigiContainerIteratorAdaptor<IndexType,DigiType>::type 
          DigiRangeIterator;
  
  DigiRangeIterator begin() const { 
    return 
      DigiContainerIteratorAdaptor<IndexType,DigiType>::adapt(data_.begin());}
  
  DigiRangeIterator end() const {
    return 
      DigiContainerIteratorAdaptor<IndexType,DigiType>::adapt(data_.end());}
  
  
private:

  typedef  std::map<IndexType,std::vector<DigiType> > container;  
  container data_;
  

}; // MuonDigiCollection



#endif

