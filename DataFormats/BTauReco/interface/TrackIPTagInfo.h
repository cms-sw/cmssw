#ifndef BTauReco_TrackIpTagInfo_h
#define BTauReco_TrackIpTagInfo_h

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

namespace reco {
 
class TrackIPTagInfo : public JTATagInfo
 {
  public:
 struct TrackIPData
 {
  Measurement1D ip2d;
  Measurement1D ip3d;
  GlobalPoint closestToJetAxis;
  GlobalPoint closestToFirstTrack;
  float  distanceToJetAxis;
  float  distanceToFirstTrack;
 };
  
 TrackIPTagInfo(
    std::vector<TrackIPData> ipData,
   std::vector<float> prob2d,
   std::vector<float> prob3d,
   edm::RefVector<TrackCollection> selectedTracks,const JetTracksAssociationRef & jtaRef,
   const edm::Ref<VertexCollection> & pv) : JTATagInfo(jtaRef),
     m_data(ipData),  m_prob2d(prob2d),
     m_prob3d(prob3d), m_selectedTracks(selectedTracks), m_pv(pv) {}

  TrackIPTagInfo() {}
  
  virtual ~TrackIPTagInfo() {}
  
  /// clone
  virtual TrackIPTagInfo * clone(void) const {
    return new TrackIPTagInfo(*this);
  }

 /**
   Check if probability information is globally available 
   impact parameters in the collection

   Even if true for some tracks it is possible that a -1 probability is returned 
   if some problem occured
  */

  virtual bool hasProbabilities() const { return  m_data.size()==m_prob3d.size(); }
  
  /**
   Vectors of TrackIPData orderd as the selectedTracks()
   */
  const std::vector<TrackIPData> & impactParameterData() const {return m_data; }
  /**
   Return the vector of tracks for which the IP information is available
   Quality cuts are applied to reject fake tracks  
  */
  const edm::RefVector<TrackCollection> & selectedTracks() const { return m_selectedTracks; }
  const std::vector<float> & probabilities(int ip) const {return (ip==0)?m_prob3d:m_prob2d; }

  typedef enum {IP3DSig = 0, Prob3D = 1, IP2DSig = 2, Prob2D =3 , 
                IP3DValue =4, IP2DValue=5 } SortCriteria;

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
   Return the list of track index sorted by mode
  */ 
  std::vector<size_t> sortedIndexes(SortCriteria mode = IP3DSig) const;
  reco::TrackRefVector sortedTracks(std::vector<size_t> indexes) const;

  virtual TaggingVariableList taggingVariables(void) const; 
 
  edm::Ref<VertexCollection>   primaryVertex() const {return m_pv; }
   private:
   std::vector<TrackIPData> m_data;
   std::vector<float> m_prob2d;   
   std::vector<float> m_prob3d;   
   edm::RefVector<TrackCollection> m_selectedTracks;
   edm::Ref<VertexCollection> m_pv;

};

//typedef edm::ExtCollection< TrackIPTagInfo,JetTagCollection> TrackCountingExtCollection;
//typedef edm::OneToOneAssociation<JetTagCollection, TrackIPTagInfo> TrackCountingExtCollection;

DECLARE_EDM_REFS( TrackIPTagInfo )

}

#endif
