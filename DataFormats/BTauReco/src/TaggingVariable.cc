#include <algorithm>
#include <utility>
#include <vector>
using namespace std;

#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace reco {

const char* TaggingVariableDescription[] = {
  /* [jetEnergy]                                = */  "jet energy",
  /* [jetPt]                                    = */  "jet transverse momentum",
  /* [jetEta]                                   = */  "jet pseudorapidity",
  /* [jetPhi]                                   = */  "jet polar angle",
  /* [trackMomentum]                            = */  "track momentum",
  /* [trackEta]                                 = */  "track pseudorapidity",
  /* [trackPhi]                                 = */  "track polar angle",
  /* [trackip2d]                                = */  "track 2D impact parameter significance",
  /* [trackSip2d]                               = */  "track 2D signed impact parameter significance",
  /* [trackSip3d]                               = */  "track 3D signed impact parameter significance",
  /* [trackPtRel]                               = */  "track transverse momentum, relative to the jet axis",
  /* [trackPpar]                                = */  "track parallel momentum, along the jet axis",
  /* [trackEtaRel]                              = */  "track pseudorapidity, relative to the jet axis",
  /* [trackDeltaR]                              = */  "track pseudoangular distance from the jet axis",
  /* [trackPtRatio]                             = */  "track transverse momentum, relative to the jet axis, normalized to its energy",
  /* [trackPparRatio]                           = */  "track parallel momentum, along the jet axis, normalized to its energy",
  /* [vertexCategory]                           = */  "category of secondary vertex (Reco, Pseudo, No)",
  /* [vertexMass]                               = */  "mass of secondary vertex",
  /* [vertexMultiplicity]                       = */ "track multiplicity at secondary vertex",
  /* [flightDistance2DSignificance]             = */ "significance in 2d of distance between primary and secondary vtx",
  /* [flightDistance3DSignificance]             = */ "significance in 3d of distance between primary and secondary vtx",
  /* [secondaryVtxEnergyRatio]                  = */ "ratio of energy at secondary vertex over total energy",
  /* [piontracksEtjetEtRatio]                   = */ "ratio of pion tracks transverse energy over jet energy",
  /* [trackSip2dAbCharm]                        = */ "track 2D signed impact parameter significance above charm mass",
  /* [neutralEnergy]                            = */ "neutral ECAL clus. energy sum",
  /* [neutralEnergyOverCombinedEnergy]          = */ "neutral ECAL clus. energy sum/(neutral ECAL clus. energy sum + pion tracks energy)",
  /* [neutralIsolEnergy]                        = */ "neutral ECAL clus. energy sum in isolation band",
  /* [neutralIsolEnergyOverCombinedEnergy]      = */ "neutral ECAL clus. energy sum in isolation band/(neutral ECAL clus. energy sum + pion tracks energy)",
  /* [neutralEnergyRatio]                       = */ "ratio of neutral ECAL clus. energy sum in isolation band over neutral ECAL clus. energy sum",
  /* [neutralclusterNumber]                     = */ "number of neutral ECAL clus.",
  /* [neutralclusterRadius]                     = */ "mean DR between neutral ECAL clus. and lead.track",
  /* [secondaryVtxWeightedEnergyRatio]          = */ "ratio of weighted energy at secondary vertex over total energy",
  /* [jetNVertices]                             = */ "number of vertices found in a jet",
  
  /* [lastTaggingVariable]                      = */ ""
};

const char* TaggingVariableTokens[] = {
  /* [jetEnergy]                                = */ "jetEnergy",
  /* [jetPt]                                    = */ "jetPt",
  /* [jetEta]                                   = */ "jetEta",
  /* [jetPhi]                                   = */ "jetPhi",
  /* [trackMomentum]                            = */ "trackMomentum",
  /* [trackEta]                                 = */ "trackEta",
  /* [trackPhi]                                 = */ "trackPhi",
  /* [trackip2d]                                = */ "trackip2d",
  /* [trackSip2d]                               = */ "trackSip2d",
  /* [trackSip3d]                               = */ "trackSip3d",
  /* [trackPtRel]                               = */ "trackPtRel",
  /* [trackPpar]                                = */ "trackPpar",
  /* [trackEtaRel]                              = */ "trackEtaRel",
  /* [trackDeltaR]                              = */ "trackDeltaR",
  /* [trackPtRatio]                             = */ "trackPtRatio",
  /* [trackPparRatio]                           = */ "trackPparRatio",
  /* [vertexCategory]                           = */ "vertexCategory",
  /* [vertexMass]                               = */ "vertexMass",
  /* [vertexMultiplicity]                       = */ "vertexMultiplicity",
  /* [flightDistance2DSignificance]             = */ "flightDistance2DSignificance",
  /* [flightDistance3DSignificance]             = */ "flightDistance3DSignificance",
  /* [secondaryVtxEnergyRatio]                  = */ "secondaryVtxEnergyRatio",
  /* [piontracksEtjetEtRatio]                   = */ "piontracksEtjetEtRatio",
  /* [trackSip2dAbCharm]                        = */ "trackSip2dAbCharm",
  /* [neutralEnergy]                            = */ "neutralEnergy",
  /* [neutralEnergyOverCombinedEnergy]          = */ "neutralEnergyOverCombinedEnergy",
  /* [neutralIsolEnergy]                        = */ "neutralIsolEnergy",
  /* [neutralIsolEnergyOverCombinedEnergy]      = */ "neutralIsolEnergyOverCombinedEnergy",
  /* [neutralEnergyRatio]                       = */ "neutralEnergyRatio",
  /* [neutralclusterNumber]                     = */ "neutralclusterNumber",
  /* [neutralclusterRadius]                     = */ "neutralclusterRadius",
  /* [secondaryVtxWeightedEnergyRatio]          = */ "secondaryVtxWeightedEnergyRatio",
  /* [jetNVertices]                             = */ "jetNVertices",

  /* [lastTaggingVariable]                      = */ "lastTaggingVariable"
};

btau::TaggingVariableName getTaggingVariableName( const std::string & name )
{
  for (int i = 0; i <= reco::btau::lastTaggingVariable; i++)
    if (name == TaggingVariableTokens[i])
      return (reco::btau::TaggingVariableName) (i);
  return btau::lastTaggingVariable;
}

// check if a tag is present in the TaggingVariableList
bool TaggingVariableList::checkTag( TaggingVariableName tag ) const {
  return binary_search( m_list.begin(), m_list.end(), tag, TaggingVariableCompare() );
}

void TaggingVariableList::insert( const TaggingVariable & variable, bool delayed /* = false */ ) {
  m_list.push_back( variable );
  if (not delayed) finalize();
}

void TaggingVariableList::insert( TaggingVariableName tag, TaggingValue value, bool delayed /* = false */ ) {
  m_list.push_back( TaggingVariable( tag, value ) );
  if (not delayed) finalize();
}

void TaggingVariableList::insert( TaggingVariableName tag, const vector<TaggingValue> & values, bool delayed /* = false */ ) {
  for (vector<TaggingValue>::const_iterator i = values.begin(); i != values.end(); i++) {
    m_list.push_back( TaggingVariable(tag, *i) );
  }
  if (not delayed) finalize();
}

void TaggingVariableList::insert( const TaggingVariableList & list ) {
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
