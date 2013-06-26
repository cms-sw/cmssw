#ifndef AlignmentSurfaceDeformations_H
#define AlignmentSurfaceDeformations_H

#include <vector>

#include "CondFormats/Alignment/interface/Definitions.h"

/// \class AlignmentSurfaceDeformations
///
/// Class for DB storage of surface deformation parameters. The actual
/// parameters for all detector IDs are stored inside one big vector.
/// Access is provided via a pair of iterators for this vector.
///
///  $Date: 2010/10/29 10:09:35 $
///  $Revision: 1.1 $
/// (last update by $Author: mussgill $)

class AlignmentSurfaceDeformations {
public:

  struct Item {
    align::ID m_rawId;
    int m_parametrizationType;
    int m_index;
  };
  
  typedef std::vector<Item> ItemVector;
  typedef std::vector<align::Scalar>::const_iterator ParametersConstIterator;
  typedef std::pair<ParametersConstIterator,ParametersConstIterator> ParametersConstIteratorPair;

  AlignmentSurfaceDeformations() { }
  virtual ~AlignmentSurfaceDeformations() { }
  
  /// Test of empty vector without having to look into internals:
  inline bool empty() const { return m_items.empty(); }
  
  /// Add a new item
  bool add(align::ID rawId, int type, const std::vector<align::Scalar> & parameters) {

    Item item;
    item.m_rawId = rawId;
    item.m_parametrizationType = type;
    item.m_index = m_parameters.size();
    m_items.push_back(item);

    m_parameters.reserve(m_parameters.size() + parameters.size());
    std::copy(parameters.begin(), parameters.end(), std::back_inserter(m_parameters));

    return true;
  }

  /// Get vector of all items
  const ItemVector & items() const {return m_items; }

  /// Get a pair of iterators for the item at given index. The iterators can
  /// be used to access the actual parameters for that item
  ParametersConstIteratorPair parameters( size_t index ) const {
    ParametersConstIteratorPair pair;
    pair.first  = m_parameters.begin() + m_items[index].m_index;
    if (index<m_items.size()-1) {
      pair.second = m_parameters.begin() + m_items[index+1].m_index;
    } else {
      pair.second = m_parameters.end();
    }
    return pair;
  }
  
 private:

  std::vector<align::Scalar> m_parameters;
  ItemVector m_items;
};

#endif // AlignmentSurfaceDeformations_H
