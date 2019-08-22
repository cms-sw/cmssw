#ifndef DataFormats_BTauReco_IpTagInfo_h
#define DataFormats_BTauReco_IpTagInfo_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include <cmath>
#include <map>
#include <Math/VectorUtil.h>
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

namespace reco {
  namespace btag {

    inline const reco::Track* toTrack(const reco::TrackBaseRef& t) { return &(*t); }
    inline const reco::Track* toTrack(const reco::TrackRef& t) { return &(*t); }
    inline const reco::Track* toTrack(const reco::CandidatePtr& c) { return (*c).bestTrack(); }

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
      double min_pT, max_pT;
      double min_pT_dRcut, max_pT_dRcut;
      double max_pT_trackPTcut;
    };
    enum SortCriteria { IP3DSig = 0, Prob3D, IP2DSig, Prob2D, IP3DValue, IP2DValue };

  }  // namespace btag

  template <class Container, class Base>
  class IPTagInfo : public Base {
  public:
    typedef Container input_container;
    typedef Base base_class;

    IPTagInfo(const std::vector<btag::TrackIPData>& ipData,
              const std::vector<float>& prob2d,
              const std::vector<float>& prob3d,
              const Container& selected,
              const Base& base,
              const edm::Ref<VertexCollection>& pv,
              const GlobalVector& axis,
              const TrackRef& ghostTrack)
        : Base(base),
          m_data(ipData),
          m_prob2d(prob2d),
          m_prob3d(prob3d),
          m_selected(selected),
          m_pv(pv),
          m_axis(axis),
          m_ghostTrack(ghostTrack) {}

    IPTagInfo() {}

    ~IPTagInfo() override {}

    /// clone
    IPTagInfo* clone(void) const override { return new IPTagInfo(*this); }

    /**
   Check if probability information is globally available 
   impact parameters in the collection

   Even if true for some tracks it is possible that a -1 probability is returned 
   if some problem occured
  */

    virtual bool hasProbabilities() const { return m_data.size() == m_prob3d.size(); }

    /**
   Vectors of TrackIPData orderd as the selectedTracks()
   */
    const std::vector<btag::TrackIPData>& impactParameterData() const { return m_data; }

    /**
   Return the vector of tracks for which the IP information is available
   Quality cuts are applied to reject fake tracks  
  */
    const Container& selected() const { return m_selected; }

    //legacy name for compatibility
    const Container& selectedTracks() const { return m_selected; }

    const std::vector<float>& probabilities(int ip) const { return (ip == 0) ? m_prob3d : m_prob2d; }

    /**
   Return the list of track index sorted by mode
   A cut can is specified to select only tracks with
   IP value or significance > cut 
   or
   probability < cut
   (according to the specified mode)
  */
    std::vector<size_t> sortedIndexesWithCut(float cut, btag::SortCriteria mode = reco::btag::IP3DSig) const;

    /**
   variable jet-to track association:
   returns vector of bool, indicating for each track whether it passed 
   the variable JTA.
  */
    std::vector<bool> variableJTA(const btag::variableJTAParameters& params) const;
    static bool passVariableJTA(const btag::variableJTAParameters& params,
                                double jetpt,
                                double trackpt,
                                double jettrackdr);

    /**
   Return the list of track index sorted by mode
  */
    std::vector<size_t> sortedIndexes(btag::SortCriteria mode = reco::btag::IP3DSig) const;
    Container sorted(const std::vector<size_t>& indexes) const;
    Container sortedTracks(const std::vector<size_t>& indexes) const { return sorted(indexes); }

    TaggingVariableList taggingVariables(void) const override;

    const edm::Ref<VertexCollection>& primaryVertex() const { return m_pv; }

    const GlobalVector& axis() const { return m_axis; }
    const TrackRef& ghostTrack() const { return m_ghostTrack; }

    const Track* selectedTrack(size_t i) const { return reco::btag::toTrack(m_selected[i]); }

    // Used by ROOT storage
    CMS_CLASS_VERSION(11)

  private:
    std::vector<btag::TrackIPData> m_data;
    std::vector<float> m_prob2d;
    std::vector<float> m_prob3d;
    Container m_selected;
    edm::Ref<VertexCollection> m_pv;
    GlobalVector m_axis;
    TrackRef m_ghostTrack;
  };

  //Explicit templates:
  //template <> const Track *IPTagInfo<TrackRefVector,JTATagInfo>::selectedTrack(size_t i) const {return &(*m_selected[i]);}
  //template <> const Track *IPTagInfo<std::vector<CandidatePtr>,BaseTagInfo>::selectedTrack(size_t i) const {return (*m_selected[i]).bestTrack();}

  template <class Container, class Base>
  TaggingVariableList IPTagInfo<Container, Base>::taggingVariables(void) const {
    TaggingVariableList vars;

    math::XYZVector jetDir = Base::jet()->momentum().Unit();
    bool havePv = primaryVertex().isNonnull();
    GlobalPoint pv;
    if (havePv)
      pv = GlobalPoint(primaryVertex()->x(), primaryVertex()->y(), primaryVertex()->z());

    std::vector<size_t> indexes = sortedIndexes();  // use default criterium
    for (std::vector<size_t>::const_iterator it = indexes.begin(); it != indexes.end(); ++it) {
      using namespace ROOT::Math;
      const Track* track = selectedTrack(*it);
      const btag::TrackIPData* data = &m_data[*it];
      const math::XYZVector& trackMom = track->momentum();
      double trackMag = std::sqrt(trackMom.Mag2());

      vars.insert(btau::trackMomentum, trackMag, true);
      vars.insert(btau::trackEta, trackMom.Eta(), true);
      vars.insert(btau::trackEtaRel, reco::btau::etaRel(jetDir, trackMom), true);
      vars.insert(btau::trackPtRel, VectorUtil::Perp(trackMom, jetDir), true);
      vars.insert(btau::trackPPar, jetDir.Dot(trackMom), true);
      vars.insert(btau::trackDeltaR, VectorUtil::DeltaR(trackMom, jetDir), true);
      vars.insert(btau::trackPtRatio, VectorUtil::Perp(trackMom, jetDir) / trackMag, true);
      vars.insert(btau::trackPParRatio, jetDir.Dot(trackMom) / trackMag, true);
      vars.insert(btau::trackSip3dVal, data->ip3d.value(), true);
      vars.insert(btau::trackSip3dSig, data->ip3d.significance(), true);
      vars.insert(btau::trackSip2dVal, data->ip2d.value(), true);
      vars.insert(btau::trackSip2dSig, data->ip2d.significance(), true);
      vars.insert(btau::trackDecayLenVal, havePv ? (data->closestToJetAxis - pv).mag() : -1.0, true);
      vars.insert(btau::trackJetDistVal, data->distanceToJetAxis.value(), true);
      vars.insert(btau::trackJetDistSig, data->distanceToJetAxis.significance(), true);
      vars.insert(btau::trackGhostTrackDistVal, data->distanceToGhostTrack.value(), true);
      vars.insert(btau::trackGhostTrackDistSig, data->distanceToGhostTrack.significance(), true);
      vars.insert(btau::trackGhostTrackWeight, data->ghostTrackWeight, true);
      vars.insert(btau::trackChi2, track->normalizedChi2(), true);
      vars.insert(btau::trackNTotalHits, track->hitPattern().numberOfValidHits(), true);
      vars.insert(btau::trackNPixelHits, track->hitPattern().numberOfValidPixelHits(), true);
    }
    vars.finalize();
    return vars;
  }

  template <class Container, class Base>
  Container IPTagInfo<Container, Base>::sorted(const std::vector<size_t>& indexes) const {
    Container tr;
    for (size_t i = 0; i < indexes.size(); i++)
      tr.push_back(m_selected[indexes[i]]);
    return tr;
  }

  template <class Container, class Base>
  std::vector<bool> IPTagInfo<Container, Base>::variableJTA(const btag::variableJTAParameters& params) const {
    std::vector<bool> result;

    //Jet parameters
    double jetpT = Base::jet()->pt();
    math::XYZVector jetDir = Base::jet()->momentum().Unit();

    for (size_t i = 0; i < m_selected.size(); i++) {
      //Track parameters
      const Track* track = selectedTrack(i);
      double trackpT = track->pt();
      const math::XYZVector& trackMom = track->momentum();

      // do the math in passVariableJTA
      result.push_back(passVariableJTA(params, jetpT, trackpT, ROOT::Math::VectorUtil::DeltaR(trackMom, jetDir)));
    }

    return result;
  }

  template <class Container, class Base>
  std::vector<size_t> IPTagInfo<Container, Base>::sortedIndexes(btag::SortCriteria mode) const {
    using namespace reco::btag;
    float cut = -1e99;
    if ((mode == Prob3D || mode == Prob2D))
      cut = 1e99;
    return sortedIndexesWithCut(cut, mode);
  }

  template <class Container, class Base>
  std::vector<size_t> IPTagInfo<Container, Base>::sortedIndexesWithCut(float cut, btag::SortCriteria mode) const {
    std::multimap<float, size_t> sortedIdx;
    size_t nSelectedTracks = m_selected.size();
    std::vector<size_t> result;
    using namespace reco::btag;

    //check if probabilities are available
    if ((mode == Prob3D || mode == Prob2D) && !hasProbabilities()) {
      return result;
    }

    for (size_t i = 0; i < nSelectedTracks; i++) {
      float sortingKey;
      switch (mode) {
        case IP3DSig:
          sortingKey = m_data[i].ip3d.significance();
          break;
        case IP2DSig:
          sortingKey = m_data[i].ip2d.significance();
          break;
        case IP3DValue:
          sortingKey = m_data[i].ip3d.value();
          break;
        case IP2DValue:
          sortingKey = m_data[i].ip2d.value();
          break;
        case Prob3D:
          sortingKey = m_prob3d[i];
          break;
        case Prob2D:
          sortingKey = m_prob2d[i];
          break;

        default:
          sortingKey = i;
      }
      sortedIdx.insert(std::pair<float, size_t>(sortingKey, i));
    }

    //Descending:
    if (mode == IP3DSig || mode == IP2DSig || mode == IP3DValue || mode == IP2DValue) {
      for (std::multimap<float, size_t>::reverse_iterator it = sortedIdx.rbegin(); it != sortedIdx.rend(); it++)
        if (it->first >= cut)
          result.push_back(it->second);
    } else
    //Ascending:
    {
      for (std::multimap<float, size_t>::iterator it = sortedIdx.begin(); it != sortedIdx.end(); it++)
        if (it->first <= cut)
          result.push_back(it->second);
    }
    return result;
  }

  template <class Container, class Base>
  bool IPTagInfo<Container, Base>::passVariableJTA(const btag::variableJTAParameters& params,
                                                   double jetpT,
                                                   double trackpT,
                                                   double jettrackdR) {
    bool pass = false;

    // intermediate pt range (between min_pT and max_pT), apply variable JTA !
    if (jetpT > params.min_pT && jetpT < params.max_pT) {
      double deltaRfunction_highpt = -jetpT * params.a_dR + params.b_dR;
      double ptfunction_highpt = jetpT * params.a_pT + params.b_pT;

      if (jettrackdR < deltaRfunction_highpt && trackpT > ptfunction_highpt)
        pass = true;

      //  cout << "IPTagInfo: passVariableJTA: dR and TrackpT " << jettrackdR << " " << trackpT << endl;

      //high pt range, apply fixed default cuts
    } else if (jetpT > params.max_pT) {
      if (jettrackdR < params.max_pT_dRcut && trackpT > params.max_pT_trackPTcut)
        pass = true;

      // low pt range, apply fixed default cuts
    } else {
      if (jettrackdR < params.min_pT_dRcut)
        pass = true;
    }

    return pass;
  }
}  // namespace reco
#endif
