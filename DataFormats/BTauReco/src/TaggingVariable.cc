#include <algorithm>
#include <utility>
#include <vector>
using namespace std;

#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace reco {

const char* TaggingVariableDescription[] = {
  /* [jetEnergy]      = */  "jet energy",
  /* [jetEta]         = */  "jet pseudorapidity",
  /* [jetPhi]         = */  "jet polar angle",
  /* [trackMomentum]  = */  "track momentum",
  /* [trackEta]       = */  "track pseudorapidity",
  /* [trackPhi]       = */  "track polar angle",
  /* [trackSip2d]     = */  "track 2D signed impact parameter significance",
  /* [trackSip3d]     = */  "track 3D signed impact parameter significance",
  /* [trackPtRel]     = */  "track transverse momentum, relative to the jet axis",
  /* [trackPpar]      = */  "track parallel momentum, along the jet axis",
  /* [trackEtaRel]    = */  "track pseudorapidity, relative to the jet axis",
  /* [trackDeltaR]    = */  "track pseudoangular distance from the jet axis",
  /* [trackPtRatio]   = */  "track transverse momentum, relative to the jet axis, normalized to its energy",
  /* [trackPparRatio] = */  "track parallel momentum, along the jet axis, normalized to its energy",

  /* [lastTaggingVariable] = */ ""
};

// check if a tag is present in the TaggingVariableList
bool TaggingVariableList::checkTag( TaggingVariableName tag ) const {
  return binary_search( m_list.begin(), m_list.end(), tag, TaggingVariableCompare() );
}

void TaggingVariableList::insert( const TaggingVariable& variable ) {
  m_list.push_back( variable );
  stable_sort( m_list.begin(), m_list.end(), TaggingVariableCompare() );
}

void TaggingVariableList::insert( TaggingVariableName tag, const vector<TaggingValue> values ) {
  for (vector<TaggingValue>::const_iterator i = values.begin(); i != values.end(); i++) {
    m_list.push_back( TaggingVariable(tag, *i) );
  }
  stable_sort( m_list.begin(), m_list.end(), TaggingVariableCompare() );
}

void TaggingVariableList::insert( const TaggingVariableList& list ) {
  vector<TaggingVariable>::size_type size = m_list.size();
  m_list.insert( m_list.end(), list.m_list.begin(), list.m_list.end() );
  inplace_merge( m_list.begin(), m_list.begin() + size, m_list.end(), TaggingVariableCompare() );
}

TaggingValue TaggingVariableList::get( TaggingVariableName tag ) const {
  if (! checkTag( tag ))
    throw edm::Exception( edm::errors::InvalidReference )
                  << "TaggingVariable " << tag << " is not present in the collection";
  
  return lower_bound( m_list.begin(), m_list.end(), tag, TaggingVariableCompare() )->second;
}

vector<TaggingValue> TaggingVariableList::getList( TaggingVariableName tag ) const {
  if (! checkTag( tag ))
    throw edm::Exception( edm::errors::InvalidReference )
                  << "TaggingVariable " << tag << " is not present in the collection";
  
  vector<TaggingValue> list;
  pair< vector<TaggingVariable>::const_iterator, vector<TaggingVariable>::const_iterator > range = 
    equal_range(m_list.begin(), m_list.end(), tag, TaggingVariableCompare() );
  
  for (vector<TaggingVariable>::const_iterator i = range.first; i != range.second; i++)
    list.push_back( i->second );

  return list;
}

} // namespace reco
