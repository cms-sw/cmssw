#ifndef RecoTauTag_RecoTau_RecoTauVertexAssociator_h
#define RecoTauTag_RecoTau_RecoTauVertexAssociator_h

/* RecoTauVertexAssociator
 *
 * Authors: Evan K. Friis, Christian Veelken, UC Davis
 *          Michalis Bachtis, UW Madison
 *
 * The associatedVertex member function retrieves the vertex from the event
 * associated to a given tau.  This class is configured using a cms.PSet.
 *
 * The required arguments are:
 *  o primaryVertexSrc - InputTag with the vertex collection
 *  o useClosestPV - Use the "closest to lead track in z" to find the vertex.
 *
 * The setEvent method must be called at least once per event.
 *
 */

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"

// histograms
#ifdef EDM_ML_DEBUG
  #include "FWCore/ServiceRegistry/interface/Service.h"
  #include "CommonTools/UtilAlgos/interface/TFileService.h"
  #include "TH1.h"
#endif

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include <map>

// Forward declarations
namespace edm {
  class ParameterSet;
  class Event;
}

namespace reco {
  class PFTau;
  class PFJet;
}

namespace reco { namespace tau {

class RecoTauVertexAssociator {
  public:
    enum Algorithm {
      kHighestPtInEvent,
      kClosestDeltaZ,
      kHighestWeigtForLeadTrack,
      kCombined
    };

    RecoTauVertexAssociator (const edm::ParameterSet& pset);
    virtual ~RecoTauVertexAssociator (){}
    /// Get the primary vertex associated to a given jet. Returns a null Ref if
    /// no vertex is found.
    reco::VertexRef associatedVertex(const PFJet& tau) const;
    reco::VertexRef associatedVertex(const PFJetRef jet) const;
    /// Convenience function to get the PV associated to the jet that
    /// seeded this tau.
    reco::VertexRef associatedVertex(const PFTau& tau) const;
    /// Load the vertices from the event.
    void setEvent(const edm::Event& evt);
    reco::TrackBaseRef getLeadTrack(const PFJet& jet) const;
    //    std::map<const PFJet*,reco::VertexRef> Employees;
  private:
    std::vector<reco::VertexRef> vertices_;
    edm::InputTag vertexTag_;
    Algorithm algo_;
    //PJ adding quality cuts
    RecoTauQualityCuts qcuts_;
    bool recoverLeadingTrk;

    // histograms
#ifdef EDM_ML_DEBUG
    TH1D* nTracks;
    TH1D* filteredTracks;
    TH1D* removedTracks;
    TH1D* leadingRemoved;
    TH1D* nVertex;
    TH1D* Vx_id;
    TH1D* Vx_id_LR;
    TH1D* h_dz;
    TH1D* h_dz_assoc;
#endif

};

} /* tau */ } /* reco */

#endif /* end of include guard: RecoTauTag_RecoTau_RecoTauVertexAssociator_h */
