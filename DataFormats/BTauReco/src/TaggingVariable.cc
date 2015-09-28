#include <algorithm>
#include <functional>
#include <ext/functional>
using namespace std;

#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace reco {

const char* const TaggingVariableDescription[] = {
  /* [jetEnergy]                                = */ "jet energy",
  /* [jetPt]                                    = */ "jet transverse momentum",
  /* [trackJetPt]                               = */ "track-based jet transverse momentum",
  /* [jetEta]                                   = */ "jet pseudorapidity",
  /* [jetPhi]                                   = */ "jet polar angle",
  /* [jetNTracks]                               = */ "tracks associated to jet",

  /* [trackMomentum]                            = */ "track momentum",
  /* [trackEta]                                 = */ "track pseudorapidity",
  /* [trackPhi]                                 = */ "track polar angle",

  /* [trackPtRel]                               = */ "track transverse momentum, relative to the jet axis",
  /* [trackPPar]                                = */ "track parallel momentum, along the jet axis",
  /* [trackEtaRel]                              = */ "track pseudorapidity, relative to the jet axis",
  /* [trackDeltaR]                              = */ "track pseudoangular distance from the jet axis",
  /* [trackPtRatio]                             = */ "track transverse momentum, relative to the jet axis, normalized to its energy",
  /* [trackPParRatio]                           = */ "track parallel momentum, along the jet axis, normalized to its energy",

  /* [trackSip2dVal]                            = */ "track 2D signed impact parameter",
  /* [trackSip2dSig]                            = */ "track 2D signed impact parameter significance",
  /* [trackSip3dVal]                            = */ "track 3D signed impact parameter",
  /* [trackSip3dSig]                            = */ "track 3D signed impact parameter significance",
  /* [trackDecayLenVal]                         = */ "track decay length",
  /* [trackDecayLenSig]                         = */ "track decay length significance",
  /* [trackJetDistVal]                          = */ "minimum track approach distance to jet axis",
  /* [trackJetDistSig]                          = */ "minimum track approach distance to jet axis signifiance",
  /* [trackGhostTrackDistVal]                   = */ "minimum approach distance to ghost track",
  /* [trackGhostTrackDistSig]                   = */ "minimum approach distance to ghost track significance",
  /* [trackGhostTrackWeight]                    = */ "weight of track participation in ghost track fit",

  /* [trackSumJetEtRatio]                       = */ "ratio of track sum transverse energy over jet energy",
  /* [trackSumJetDeltaR]                        = */ "pseudoangular distance between jet axis and track fourvector sum",

  /* [vertexCategory]                           = */ "category of secondary vertex (Reco, Pseudo, No)",
  /* [vertexLeptonCategory]                     = */ "category of secondary vertex & soft lepton (RecoNo, PseudoNo, NoNo, RecoMu, PseudoMu, NoMu, RecoEl, PseudoEl, NoEl)",

  /* [jetNSecondaryVertices]                    = */ "number of reconstructed possible secondary vertices in jet",
  /* [jetNSingleTrackVertices]                  = */ "number of single-track ghost-track vertices",

  /* [vertexMass]                               = */ "mass of track sum at secondary vertex",
  /* [vertexNTracks]                            = */ "number of tracks at secondary vertex",
  /* [vertexFitProb]                            = */ "vertex fit probability",

  /* [vertexEnergyRatio]                        = */ "ratio of energy at secondary vertex over total energy",
  /* [vertexJetDeltaR]                          = */ "pseudoangular distance between jet axis and secondary vertex direction",

  /* [flightDistance2dVal]                      = */ "transverse distance between primary and secondary vertex",
  /* [flightDistance2dSig]                      = */ "transverse distance significance between primary and secondary vertex",
  /* [flightDistance3dVal]                      = */ "distance between primary and secondary vertex",
  /* [flightDistance3dSig]                      = */ "distance significance between primary and secondary vertex",

  /* [trackSip2dValAboveCharm]                  = */ "track 2D signed impact parameter of first track lifting mass above charm",
  /* [trackSip2dSigAboveCharm]                  = */ "track 2D signed impact parameter significance of first track lifting mass above charm",
  /* [trackSip3dValAboveCharm]                  = */ "track 3D signed impact parameter of first track lifting mass above charm",
  /* [trackSip3dSigAboveCharm]                  = */ "track 3D signed impact parameter significance of first track lifting mass above charm",

  /* [trackP0Par]                               = */ "track momentum along the jet axis, in the jet rest frame",
  /* [trackP0ParRatio]                          = */ "track momentum along the jet axis, in the jet rest frame, normalized to its energy"
  /* [trackChi2]                                = */ "chi2 of the track fit",
  /* [trackNTotalHits]                          = */ "number of valid total hits",
  /* [trackNPixelHits]                          = */ "number of valid pixel hits",

  /* [leptonQuality]                            = */ "lepton identification quality",
  /* [leptonQuality2]                           = */ "lepton identification quality 2",

  /* [chargedHadronEnergyFraction]              = */ "fraction of the jet energy coming from charged hadrons",
  /* [neutralHadronEnergyFraction]              = */ "fraction of the jet energy coming from neutral hadrons",
  /* [photonEnergyFraction]                     = */ "fraction of the jet energy coming from photons",
  /* [electronEnergyFraction]                   = */ "fraction of the jet energy coming from electrons",
  /* [muonEnergyFraction]                       = */ "fraction of the jet energy coming from muons",
  /* [chargedHadronMultiplicity]                = */ "number of charged hadrons in the jet",
  /* [neutralHadronMultiplicity]                = */ "number of neutral hadrons in the jet",
  /* [photonMultiplicity]                       = */ "number of photons in the jet",
  /* [electronMultiplicity]                     = */ "number of electrons in the jet",
  /* [muonMultiplicity]                         = */ "number of muons in the jet",
  /* [hadronMultiplicity]                       = */ "total number of charged and neutral hadrons in the jet",
  /* [hadronPhotonMultiplicity]                 = */ "total number of photons, charged and neutral hadrons in the jet",
  /* [totalMultiplicity]                        = */ "total number of photons, electrons, muons, charged and neutral hadrons in the jet",

  /* [massVertexEnergyFraction]                 = */ "vertexmass times fraction of the vertex energy w.r.t. the jet energy",
  /* [vertexBoostOverSqrtJetPt]                 = */ "variable related to the boost of the vertex system in flight direction",

  /* [leptonSip2d]                              = */ "2D signed impact parameter of the soft lepton",
  /* [leptonSip3d]                              = */ "3D signed impact parameter of the soft lepton",
  /* [leptonPtRel]                              = */ "transverse momentum of the soft lepton wrt. the jet axis",
  /* [leptonP0Par]                              = */ "momentum of the soft lepton along the jet direction, in the jet rest frame",
  /* [leptonEtaRel]                             = */ "pseudo)rapidity of the soft lepton along jet axis",
  /* [leptonDeltaR]                             = */ "pseudo)angular distance of the soft lepton to jet axis",
  /* [leptonRatio],                             = */ "momentum of the soft lepton over jet energy",
  /* [leptonRatioRel]                           = */ "momentum of the soft lepton parallel to jet axis over jet energy",
  /* [electronMVA]                              = */ "mva output of the electron ID",


  /* [algoDiscriminator]                        = */ "discriminator output of an algorithm",

  /* [lastTaggingVariable]                      = */ ""
};

const char* const TaggingVariableTokens[] = {
  /* [jetEnergy]                                = */ "jetEnergy",
  /* [jetPt]                                    = */ "jetPt",
  /* [trackJetPt]                               = */ "trackJetPt",
  /* [jetEta]                                   = */ "jetEta",
  /* [jetPhi]                                   = */ "jetPhi",
  /* [jetNTracks]                               = */ "jetNTracks",

  /* [trackMomentum]                            = */ "trackMomentum",
  /* [trackEta]                                 = */ "trackEta",
  /* [trackPhi]                                 = */ "trackPhi",

  /* [trackPtRel]                               = */ "trackPtRel",
  /* [trackPPar]                                = */ "trackPPar",
  /* [trackEtaRel]                              = */ "trackEtaRel",
  /* [trackDeltaR]                              = */ "trackDeltaR",
  /* [trackPtRatio]                             = */ "trackPtRatio",
  /* [trackPParRatio]                           = */ "trackPParRatio",

  /* [trackSip2dVal]                            = */ "trackSip2dVal",
  /* [trackSip2dSig]                            = */ "trackSip2dSig",
  /* [trackSip3dVal]                            = */ "trackSip3dVal",
  /* [trackSip3dSig]                            = */ "trackSip3dSig",
  /* [trackDecayLenVal]                         = */ "trackDecayLenVal",
  /* [trackDecayLenSig]                         = */ "trackDecayLenSig",
  /* [trackJetDistVal]                          = */ "trackJetDist",    //FIXME
  /* [trackJetDistSig]                          = */ "trackJetDistSig",
  /* [trackGhostTrackDistVal]                   = */ "trackGhostTrackDistVal",
  /* [trackGhostTrackDistSig]                   = */ "trackGhostTrackDistSig",
  /* [trackGhostTrackWeight]                    = */ "trackGhostTrackWeight",

  /* [trackSumJetEtRatio]                       = */ "trackSumJetEtRatio",
  /* [trackSumJetDeltaR]                        = */ "trackSumJetDeltaR",

  /* [vertexCategory]                           = */ "vertexCategory",
  /* [vertexLeptonCategory]                     = */ "vertexLeptonCategory",

  /* [jetNSecondaryVertices]                    = */ "jetNSecondaryVertices",
  /* [jetNSingleTrackVertices]                  = */ "jetNSingleTrackVertices",

  /* [vertexMass]                               = */ "vertexMass",
  /* [vertexNTracks]                            = */ "vertexNTracks",
  /* [vertexFitProb]                            = */ "vertexFitProb",

  /* [vertexEnergyRatio]                        = */ "vertexEnergyRatio",
  /* [vertexJetDeltaR]                          = */ "vertexJetDeltaR",

  /* [flightDistance2dVal]                      = */ "flightDistance2dVal",
  /* [flightDistance2dSig]                      = */ "flightDistance2dSig",
  /* [flightDistance3dVal]                      = */ "flightDistance3dVal",
  /* [flightDistance3dSig]                      = */ "flightDistance3dSig",

  /* [trackSip2dValAboveCharm]                  = */ "trackSip2dValAboveCharm",
  /* [trackSip2dSigAboveCharm]                  = */ "trackSip2dSigAboveCharm",
  /* [trackSip3dValAboveCharm]                  = */ "trackSip3dValAboveCharm",
  /* [trackSip3dSigAboveCharm]                  = */ "trackSip3dSigAboveCharm",

  /* [leptonQuality]                            = */ "leptonQuality",
  /* [leptonQuality2]                           = */ "leptonQuality2",

  /* [trackP0Par]                               = */ "trackP0Par",
  /* [trackP0ParRatio]                          = */ "trackP0ParRatio",
  /* [trackChi2]                                = */ "trackChi2",
  /* [trackNTotalHits]                          = */ "trackNTotalHits",
  /* [trackNPixelHits]                          = */ "trackNPixelHits",


  /* [chargedHadronEnergyFraction]              = */ "chargedHadronEnergyFraction",
  /* [neutralHadronEnergyFraction]              = */ "neutralHadronEnergyFraction",
  /* [photonEnergyFraction]                     = */ "photonEnergyFraction",
  /* [electronEnergyFraction]                   = */ "electronEnergyFraction",
  /* [muonEnergyFraction],                      = */ "muonEnergyFraction",
  /* [chargedHadronMultiplicity],               = */ "chargedHadronMultiplicity",
  /* [neutralHadronMultiplicity],               = */ "neutralHadronMultiplicity",
  /* [photonMultiplicity]                       = */ "photonMultiplicity",
  /* [electronMultiplicity]                     = */ "electronMultiplicity",
  /* [muonMultiplicity],                        = */ "muonMultiplicity",
  /* [hadronMultiplicity],                      = */ "hadronMultiplicity",
  /* [hadronPhotonMultiplicity],                = */ "hadronPhotonMultiplicity",
  /* [totalMultiplicity]                        = */ "totalMultiplicity",

  /* [massVertexEnergyFraction],                = */ "massVertexEnergyFraction",
  /* [vertexBoostOverSqrtJetPt],                = */ "vertexBoostOverSqrtJetPt",
 
  /* [leptonSip2d]                              = */ "leptonSip2d",
  /* [leptonSip3d]                              = */ "leptonSip3d",
  /* [leptonPtRel]                              = */ "leptonPtRel",
  /* [leptonP0Par]                              = */ "leptonP0Par",
  /* [leptonEtaRel]                             = */ "leptonEtaRel",
  /* [leptonDeltaR]                             = */ "leptonDeltaR",
  /* [leptonRatio]                              = */ "leptonRatio",
  /* [leptonRatioRel],                          = */ "leptonRatioRel",
  /* [electronMVA]                              = */ "electronMVA",

  /* [algoDiscriminator]                        = */ "algoDiscriminator",

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

void TaggingVariableList::insert( TaggingVariableName tag, const std::vector<TaggingValue> & values, bool delayed /* = false */ ) {
  for (std::vector<TaggingValue>::const_iterator i = values.begin(); i != values.end(); i++) {
    m_list.push_back( TaggingVariable(tag, *i) );
  }
  if (not delayed) finalize();
}

void TaggingVariableList::insert( const TaggingVariableList & list ) {
  std::vector<TaggingVariable>::size_type size = m_list.size();
  m_list.insert( m_list.end(), list.m_list.begin(), list.m_list.end() );
  inplace_merge( m_list.begin(), m_list.begin() + size, m_list.end(), TaggingVariableCompare() );
}

void TaggingVariableList::finalize( void ) {
  stable_sort( m_list.begin(), m_list.end(), TaggingVariableCompare() );
}

TaggingValue TaggingVariableList::get( TaggingVariableName tag ) const {
  range r = getRange(tag);
  if (r.first == r.second)
    throw edm::Exception( edm::errors::InvalidReference )
                  << "TaggingVariable " << tag << " is not present in the collection";
  return r.first->second;
}

TaggingValue TaggingVariableList::get( TaggingVariableName tag, TaggingValue defaultValue ) const {
  range r = getRange(tag);
  if ( r.first == r.second )
    return defaultValue;
  return r.first->second;
}

std::vector<TaggingValue> TaggingVariableList::getList( TaggingVariableName tag, bool throwOnEmptyList ) const {
  using namespace __gnu_cxx;
  range r = getRange( tag );
  if ( throwOnEmptyList && r.first == r.second )
    throw edm::Exception( edm::errors::InvalidReference )
                  << "TaggingVariable " << tag << " is not present in the collection";
  std::vector<TaggingValue> list( r.second - r.first );
  transform( r.first, r.second, list.begin(), select2nd< TaggingVariable >() );
  return list;
}

TaggingVariableList::range TaggingVariableList::getRange( TaggingVariableName tag ) const {
  return equal_range( m_list.begin(), m_list.end(), tag, TaggingVariableCompare() );
}

} // namespace reco
