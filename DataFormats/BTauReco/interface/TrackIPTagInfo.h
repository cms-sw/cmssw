#ifndef BTauReco_TrackIpTagInfo_h
#define BTauReco_TrackIpTagInfo_h

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

namespace reco {
 
class TrackIPTagInfo : public JTATagInfo {
public:
  struct TrackIPData {
    GlobalPoint closestToJetAxis;
    GlobalPoint closestToGhostTrack;
    Measurement1D ip2d;
    Measurement1D ip3d;
    Measurement1D distanceToJetAxis;
    Measurement1D distanceToGhostTrack;
    float ghostTrackWeight;
  };
  struct variableJTAParameters {
    double a_dR, b_dR, a_pT, b_pT;
    double min_pT,  max_pT;
    double min_pT_dRcut,  max_pT_dRcut;
    double max_pT_trackPTcut;
  };
  
  TrackIPTagInfo(
    const std::vector<TrackIPData> & ipData,
    const std::vector<float> & prob2d,
    const std::vector<float> & prob3d,
    const edm::RefVector<TrackCollection> & selectedTracks,
    const JetTracksAssociationRef & jtaRef,
    const edm::Ref<VertexCollection> & pv,
    const GlobalVector & axis,
    const TrackRef & ghostTrack) :
      JTATagInfo(jtaRef), m_data(ipData),  m_prob2d(prob2d),
      m_prob3d(prob3d), m_selectedTracks(selectedTracks), m_pv(pv),
      m_axis(axis), m_ghostTrack(ghostTrack) {}

  TrackIPTagInfo() {}
  
  virtual ~TrackIPTagInfo() {}
  
  /// clone
  virtual TrackIPTagInfo * clone(void) const
  { return new TrackIPTagInfo(*this); }

 /**
   Check if probability information is globally available 
   impact parameters in the collection

   Even if true for some tracks it is possible that a -1 probability is returned 
   if some problem occured
  */

  virtual bool hasProbabilities() const
  { return m_data.size() == m_prob3d.size(); }
  
  /**
   Vectors of TrackIPData orderd as the selectedTracks()
   */
  const std::vector<TrackIPData> & impactParameterData() const
  { return m_data; }

  /**
   Return the vector of tracks for which the IP information is available
   Quality cuts are applied to reject fake tracks  
  */
  const edm::RefVector<TrackCollection> & selectedTracks() const { return m_selectedTracks; }
  const std::vector<float> & probabilities(int ip) const {return (ip==0)?m_prob3d:m_prob2d; }

  enum SortCriteria { IP3DSig = 0, Prob3D, IP2DSig, Prob2D, 
                      IP3DValue, IP2DValue };

  /**
   Return the list of track index sorted by mode
   A cut can is specified to select only tracks with
   IP value or significance > cut 
   or
   probability < cut
   (according to the specified mode)
  */
  std::vector<size_t> sortedIndexesWithCut(float cut, SortCriteria mode = IP3DSig) const;

  /**
   variable jet-to track association:
   returns vector of bool, indicating for each track whether it passed 
   the variable JTA.
  */
  std::vector<bool> variableJTA(const variableJTAParameters &params) const;
  static bool passVariableJTA(const variableJTAParameters &params, double jetpt, double trackpt, double jettrackdr) ;

  /**
   Return the list of track index sorted by mode
  */ 
  std::vector<size_t> sortedIndexes(SortCriteria mode = IP3DSig) const;
  reco::TrackRefVector sortedTracks(const std::vector<size_t>& indexes) const;

  virtual TaggingVariableList taggingVariables(void) const; 
 
  const edm::Ref<VertexCollection> & primaryVertex() const { return m_pv; }

  const GlobalVector & axis() const { return m_axis; }
  const TrackRef & ghostTrack() const { return m_ghostTrack; }

private:
  std::vector<TrackIPData> m_data;
  std::vector<float> m_prob2d;   
  std::vector<float> m_prob3d;   
  edm::RefVector<TrackCollection> m_selectedTracks;
  edm::Ref<VertexCollection> m_pv;
  GlobalVector m_axis;
  TrackRef m_ghostTrack;
};

//typedef edm::ExtCollection< TrackIPTagInfo,JetTagCollection> TrackCountingExtCollection;
//typedef edm::OneToOneAssociation<JetTagCollection, TrackIPTagInfo> TrackCountingExtCollection;

DECLARE_EDM_REFS( TrackIPTagInfo )

}

#endif
