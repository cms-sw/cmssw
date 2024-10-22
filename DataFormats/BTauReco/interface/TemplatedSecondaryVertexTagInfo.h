#ifndef DataFormats_BTauReco_TemplatedSecondaryVertexTagInfo_h
#define DataFormats_BTauReco_TemplatedSecondaryVertexTagInfo_h

#include <utility>
#include <vector>

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include <functional>
#include <algorithm>

#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace reco {
  namespace btag {
    inline float weight(const reco::TrackRef &t, const reco::Vertex &v) { return v.trackWeight(t); }
    inline float weight(const reco::CandidatePtr &c, const reco::VertexCompositePtrCandidate &v) {
      return std::find(v.daughterPtrVector().begin(), v.daughterPtrVector().end(), c) != v.daughterPtrVector().end();
    }

    struct TrackData {
      static constexpr int trackSelected = 0;
      static constexpr int trackUsedForVertexFit = 1;
      static constexpr int trackAssociatedToVertex = 2;
      inline bool usedForVertexFit() const { return svStatus >= trackUsedForVertexFit; }
      inline bool associatedToVertex() const { return svStatus >= trackAssociatedToVertex; }
      inline bool associatedToVertex(unsigned int index) const {
        return svStatus == trackAssociatedToVertex + (int)index;
      }
      int svStatus;
    };
    typedef std::pair<unsigned int, TrackData> IndexedTrackData;

  }  // namespace btag
  template <class IPTI, class VTX>
  class TemplatedSecondaryVertexTagInfo : public BaseTagInfo {
  public:
    typedef reco::btag::TrackData TrackData;
    typedef reco::btag::IndexedTrackData IndexedTrackData;

    struct VertexData {
      VTX vertex;
      Measurement1D dist1d, dist2d, dist3d;
      GlobalVector direction;

      // Used by ROOT storage
      CMS_CLASS_VERSION(12)
    };

    struct TrackFinder {
      TrackFinder(const typename IPTI::input_container &tracks, const typename IPTI::input_container::value_type &track)
          : tracks(tracks), track(track) {}

      bool operator()(const IndexedTrackData &idt) { return tracks[idt.first] == track; }

      const typename IPTI::input_container &tracks;
      const typename IPTI::input_container::value_type &track;
    };

    struct VertexTrackSelector {
      bool operator()(const IndexedTrackData &idt) { return idt.second.associatedToVertex(); }
    };

    struct IndexedVertexTrackSelector {
      IndexedVertexTrackSelector(unsigned int index) : index(index) {}

      bool operator()(const IndexedTrackData &idt) { return idt.second.associatedToVertex(index); }

      unsigned int index;
    };

    typedef typename IPTI::input_container input_container;

    TemplatedSecondaryVertexTagInfo() {}
    ~TemplatedSecondaryVertexTagInfo() override {}

    TemplatedSecondaryVertexTagInfo(const std::vector<IndexedTrackData> &trackData,
                                    const std::vector<VertexData> &svData,
                                    unsigned int vertexCandidates,
                                    const edm::Ref<std::vector<IPTI> > &);

    /// clone
    TemplatedSecondaryVertexTagInfo *clone(void) const override { return new TemplatedSecondaryVertexTagInfo(*this); }

    const edm::Ref<std::vector<IPTI> > &trackIPTagInfoRef() const { return m_trackIPTagInfoRef; }

    edm::RefToBase<Jet> jet(void) const override { return m_trackIPTagInfoRef->jet(); }

    //	virtual input_container ipTracks(void) const
    //	{ return m_trackIPTagInfoRef->tracks(); }

    //AR TODO is it needed?
    //	const JetTracksAssociationRef &jtaRef(void) const
    //	{ return m_trackIPTagInfoRef->jtaRef(); }

    const VTX &secondaryVertex(unsigned int index) const { return m_svData[index].vertex; }

    unsigned int nSelectedTracks() const { return m_trackData.size(); }
    unsigned int nVertexTracks() const;
    unsigned int nVertexTracks(unsigned int index) const;
    unsigned int nVertices() const { return m_svData.size(); }
    unsigned int nVertexCandidates() const { return m_vertexCandidates; }

    input_container selectedTracks() const;
    input_container vertexTracks() const;
    input_container vertexTracks(unsigned int index) const;

    typename input_container::value_type track(unsigned int index) const;
    unsigned int findTrack(const typename input_container::value_type &track) const;

    const TrackData &trackData(unsigned int index) const;
    const TrackData &trackData(const typename input_container::value_type &track) const;

    const reco::btag::TrackIPData &trackIPData(unsigned int index) const;
    const reco::btag::TrackIPData &trackIPData(const typename input_container::value_type &track) const;

    float trackWeight(unsigned int svIndex, unsigned int trackindex) const;
    float trackWeight(unsigned int svIndex, const typename input_container::value_type &track) const;

    Measurement1D flightDistance(unsigned int index, int dim = 0) const {
      if (dim == 1)
        return m_svData[index].dist1d;
      else if (dim == 2)
        return m_svData[index].dist2d;
      else
        return m_svData[index].dist3d;
    }
    const GlobalVector &flightDirection(unsigned int index) const { return m_svData[index].direction; }
    TaggingVariableList taggingVariables() const override;

    // Used by ROOT storage
    CMS_CLASS_VERSION(11)

  private:
    std::vector<IndexedTrackData> m_trackData;
    std::vector<VertexData> m_svData;
    unsigned int m_vertexCandidates;

    edm::Ref<std::vector<IPTI> > m_trackIPTagInfoRef;
  };

  template <class IPTI, class VTX>
  TemplatedSecondaryVertexTagInfo<IPTI, VTX>::TemplatedSecondaryVertexTagInfo(
      const std::vector<IndexedTrackData> &trackData,
      const std::vector<VertexData> &svData,
      unsigned int vertexCandidates,
      const edm::Ref<std::vector<IPTI> > &trackIPTagInfoRef)
      : m_trackData(trackData),
        m_svData(svData),
        m_vertexCandidates(vertexCandidates),
        m_trackIPTagInfoRef(trackIPTagInfoRef) {}

  template <class IPTI, class VTX>
  unsigned int TemplatedSecondaryVertexTagInfo<IPTI, VTX>::nVertexTracks() const {
    return std::count_if(m_trackData.begin(), m_trackData.end(), VertexTrackSelector());
  }

  template <class IPTI, class VTX>
  unsigned int TemplatedSecondaryVertexTagInfo<IPTI, VTX>::nVertexTracks(unsigned int index) const {
    return std::count_if(m_trackData.begin(), m_trackData.end(), IndexedVertexTrackSelector(index));
  }

  template <class IPTI, class VTX>
  typename reco::TemplatedSecondaryVertexTagInfo<IPTI, VTX>::input_container
  TemplatedSecondaryVertexTagInfo<IPTI, VTX>::selectedTracks() const {
    input_container trackRefs;
    const input_container &trackIPTrackRefs = m_trackIPTagInfoRef->selectedTracks();

    for (typename std::vector<
             typename reco::TemplatedSecondaryVertexTagInfo<IPTI, VTX>::IndexedTrackData>::const_iterator iter =
             m_trackData.begin();
         iter != m_trackData.end();
         iter++)

      trackRefs.push_back(trackIPTrackRefs[iter->first]);

    return trackRefs;
  }

  template <class IPTI, class VTX>
  typename TemplatedSecondaryVertexTagInfo<IPTI, VTX>::input_container
  TemplatedSecondaryVertexTagInfo<IPTI, VTX>::vertexTracks() const {
    input_container trackRefs;
    const input_container &trackIPTrackRefs = m_trackIPTagInfoRef->selectedTracks();

    for (typename std::vector<
             typename reco::TemplatedSecondaryVertexTagInfo<IPTI, VTX>::IndexedTrackData>::const_iterator iter =
             m_trackData.begin();
         iter != m_trackData.end();
         iter++)

      if (iter->second.associatedToVertex())
        trackRefs.push_back(trackIPTrackRefs[iter->first]);

    return trackRefs;
  }

  template <class IPTI, class VTX>
  typename TemplatedSecondaryVertexTagInfo<IPTI, VTX>::input_container
  TemplatedSecondaryVertexTagInfo<IPTI, VTX>::vertexTracks(unsigned int index) const {
    input_container trackRefs;
    const input_container &trackIPTrackRefs = m_trackIPTagInfoRef->selectedTracks();

    for (typename std::vector<
             typename reco::TemplatedSecondaryVertexTagInfo<IPTI, VTX>::IndexedTrackData>::const_iterator iter =
             m_trackData.begin();
         iter != m_trackData.end();
         iter++)

      if (iter->second.associatedToVertex(index))
        trackRefs.push_back(trackIPTrackRefs[iter->first]);

    return trackRefs;
  }

  template <class IPTI, class VTX>
  typename TemplatedSecondaryVertexTagInfo<IPTI, VTX>::input_container::value_type
  TemplatedSecondaryVertexTagInfo<IPTI, VTX>::track(unsigned int index) const {
    return m_trackIPTagInfoRef->selectedTracks()[m_trackData[index].first];
  }

  template <class IPTI, class VTX>
  unsigned int TemplatedSecondaryVertexTagInfo<IPTI, VTX>::findTrack(
      const typename input_container::value_type &track) const {
    typename std::vector<typename reco::TemplatedSecondaryVertexTagInfo<IPTI, VTX>::IndexedTrackData>::const_iterator
        pos = std::find_if(
            m_trackData.begin(), m_trackData.end(), TrackFinder(m_trackIPTagInfoRef->selectedTracks(), track));

    if (pos == m_trackData.end())
      throw edm::Exception(edm::errors::InvalidReference) << "Track not found in "
                                                             " TemplatedSecondaryVertexTagInfo<IPTI,VTX>::findTrack."
                                                          << std::endl;

    return pos - m_trackData.begin();
  }

  template <class IPTI, class VTX>
  const typename TemplatedSecondaryVertexTagInfo<IPTI, VTX>::TrackData &
  TemplatedSecondaryVertexTagInfo<IPTI, VTX>::trackData(unsigned int index) const {
    return m_trackData[index].second;
  }

  template <class IPTI, class VTX>
  const typename TemplatedSecondaryVertexTagInfo<IPTI, VTX>::TrackData &
  TemplatedSecondaryVertexTagInfo<IPTI, VTX>::trackData(const typename input_container::value_type &track) const {
    return m_trackData[findTrack(track)].second;
  }

  template <class IPTI, class VTX>
  const reco::btag::TrackIPData &TemplatedSecondaryVertexTagInfo<IPTI, VTX>::trackIPData(unsigned int index) const {
    return m_trackIPTagInfoRef->impactParameterData()[m_trackData[index].first];
  }

  template <class IPTI, class VTX>
  const reco::btag::TrackIPData &TemplatedSecondaryVertexTagInfo<IPTI, VTX>::trackIPData(
      const typename input_container::value_type &track) const {
    return trackIPData(findTrack(track));
  }

  template <class IPTI, class VTX>
  float TemplatedSecondaryVertexTagInfo<IPTI, VTX>::trackWeight(
      unsigned int svIndex, const typename input_container::value_type &track) const {
    return reco::btag::weight(track, m_svData[svIndex].vertex);
  }

  template <class IPTI, class VTX>
  float TemplatedSecondaryVertexTagInfo<IPTI, VTX>::trackWeight(unsigned int svIndex, unsigned int trackIndex) const {
    return trackWeight(svIndex, track(trackIndex));
  }

  template <class IPTI, class VTX>
  TaggingVariableList TemplatedSecondaryVertexTagInfo<IPTI, VTX>::taggingVariables() const {
    TaggingVariableList vars;

    for (typename std::vector<typename TemplatedSecondaryVertexTagInfo<IPTI, VTX>::VertexData>::const_iterator iter =
             m_svData.begin();
         iter != m_svData.end();
         iter++) {
      vars.insert(btau::flightDistance1dVal, iter->dist1d.value(), true);
      vars.insert(btau::flightDistance1dSig, iter->dist1d.significance(), true);
      vars.insert(btau::flightDistance2dVal, iter->dist2d.value(), true);
      vars.insert(btau::flightDistance2dSig, iter->dist2d.significance(), true);
      vars.insert(btau::flightDistance3dVal, iter->dist3d.value(), true);
      vars.insert(btau::flightDistance3dSig, iter->dist3d.significance(), true);

      vars.insert(btau::vertexJetDeltaR, Geom::deltaR(iter->direction, jet()->momentum()), true);
    }

    vars.insert(btau::jetNSecondaryVertices, m_vertexCandidates, true);
    vars.insert(btau::vertexNTracks, nVertexTracks(), true);

    vars.finalize();
    return vars;
  }

}  // namespace reco
#endif  // DataFormats_BTauReco_TemplatedSecondaryVertexTagInfo_h
