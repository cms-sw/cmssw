#ifndef DataFormats_MuonDigiCollection_h
#define DataFormats_MuonDigiCollection_h

/**
 * \file
 * Declaration of class MuonDigiCollection
 *
 * \author Stefano ARGIRO
 * \version $Id: MuonDigiCollection.h,v 1.4 2006/03/23 13:52:56 namapane Exp $
 * \date 05 Aug 2005
 */


#include <vector>
#include <map>
#include <iterator>

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

template <typename IndexType, typename DigiType>
  class DigiContainerIterator {
  public:
    typedef typename std::vector<DigiType>::const_iterator  DigiRangeIterator;
    typedef std::map<IndexType, std::vector<DigiType> >     BaseContainer;
    typedef typename BaseContainer::const_iterator          BaseIterator;

    typedef std::pair<IndexType,
		      std::pair<DigiRangeIterator,
				DigiRangeIterator> >        value_type;
    typedef value_type                                      reference;
    typedef void                                            pointer;
    typedef typename DigiRangeIterator::difference_type     difference_type;
    typedef typename DigiRangeIterator::iterator_category   iterator_category;

    DigiContainerIterator (void) {}
    DigiContainerIterator (BaseIterator i) : base_ (i) {}
    // implicit copy constructor
    // implicit assignment operator
    // implicit destructor

    DigiContainerIterator operator++ (int)
    { return DigiContainerIterator (base_++); }
    
    DigiContainerIterator &operator++ (void)
    { ++base_; return *this; }

    bool operator== (const DigiContainerIterator &x)
    { return x.base_ == base_; }

    bool operator!= (const DigiContainerIterator &x)
    { return x.base_ != base_; }

    value_type operator* (void) const
    {
      return std::make_pair(base_->first,
			    std::make_pair(base_->second.begin(), 
					   base_->second.end()));
    }

  private:
    BaseIterator base_;
  };




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
    std::vector<DigiType> &digis = data_[index];
    digis.push_back(digi);
  }
  
  /// insert a range of digis for a  given DetUnit
  void put(Range range, const IndexType& index){
    std::vector<DigiType> &digis = data_[index];
    digis.reserve (digis.size () + (range.second - range.first));
    std::copy (range.first, range.second, std::back_inserter (digis));
    
  }
 
  /// return the digis for a given DetUnit 
  Range get(const IndexType& index) const{
    typename container::const_iterator it = data_.find(index);
    if (it==data_.end()) {
      // if data_ is empty there is no other way to get an empty range
      static std::vector<DigiType> empty;
      return std::make_pair(empty.end(),empty.end());
    } 
    const std::vector<DigiType>& digis = (*it).second;
    return std::make_pair(digis.begin(),digis.end());
  }
  
  typedef DigiContainerIterator<IndexType,DigiType> DigiRangeIterator;
  
  DigiRangeIterator begin() const { 
    return data_.begin();}
  
  DigiRangeIterator end() const {
    return data_.end();}
  
  
private:

  typedef  std::map<IndexType,std::vector<DigiType> > container;  
  container data_;
  

}; // MuonDigiCollection



#endif

