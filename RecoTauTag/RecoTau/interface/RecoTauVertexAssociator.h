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

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/JetReco/interface/JetCollection.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

#include <map>

// Forward declarations
namespace edm {
  class ParameterSet;
  class Event;
}  // namespace edm

namespace reco {
  class PFTau;
  class Jet;
}  // namespace reco

namespace reco {
  namespace tau {

    class RecoTauVertexAssociator {
    public:
      enum Algorithm { kHighestPtInEvent, kClosestDeltaZ, kHighestWeigtForLeadTrack, kCombined };

      RecoTauVertexAssociator(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC);
      virtual ~RecoTauVertexAssociator(){};
      /// Get the primary vertex associated to a given jet.
      /// Returns a null Ref if no vertex is found.
      reco::VertexRef associatedVertex(const Jet& jet) const;
      /// Convenience function to get the PV associated to the jet that
      /// seeded this tau (useJet=true, old behaviour)
      /// or leaging charged hadron if set (useJet=false).
      reco::VertexRef associatedVertex(const PFTau& tau, bool useJet = false) const;
      reco::VertexRef associatedVertex(const TrackBaseRef& track) const;
      reco::VertexRef associatedVertex(const Track* track) const;

      /// Load the vertices from the event.
      void setEvent(const edm::Event& evt);
      const Track* getLeadTrack(const Jet&) const;
      const TrackBaseRef getLeadTrackRef(const Jet&) const;
      const CandidatePtr getLeadCand(const Jet&) const;

    private:
      edm::InputTag vertexTag_;
      bool vxTrkFiltering_;
      std::unique_ptr<StringCutObjectSelector<reco::Vertex>> vertexSelector_;
      std::vector<reco::VertexRef> selectedVertices_;
      std::string algorithm_;
      Algorithm algo_;
      //PJ adding quality cuts
      std::unique_ptr<RecoTauQualityCuts> qcuts_;
      bool recoverLeadingTrk_;
      enum { kLeadTrack, kLeadPFCand, kMinLeadTrackOrPFCand, kFirstTrack };
      int leadingTrkOrPFCandOption_;
      edm::EDGetTokenT<reco::VertexCollection> vxToken_;
      // containers for holding vertices associated to jets
      typedef std::map<const reco::Jet*, reco::VertexRef> JetToVtxAssoc;
      std::unique_ptr<JetToVtxAssoc> jetToVertexAssociation_;
      edm::EventNumber_t lastEvent_;
      int verbosity_;
    };

  }  // namespace tau
}  // namespace reco

#endif /* end of include guard: RecoTauTag_RecoTau_RecoTauVertexAssociator_h */
