#ifndef pfIsoCalculator_h
#define pfIsoCalculator_h

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

//template <typename Tcand>
class pfIsoCalculator {
public:
  pfIsoCalculator() { usePackedCandidates_ = true; };

  void setUsePackedCandidates(bool usePackedCandidates) { usePackedCandidates_ = usePackedCandidates; };
  void setCandidatesPacked(const edm::Handle<edm::View<pat::PackedCandidate>>& candidatesPacked) {
    candidatesView = candidatesPacked;
  };
  void setVertex(const math::XYZPoint& pv);

  template <class T, class U>
  bool isAssociatedPackedCand(const T& associatedPackedCands, const U& candidate);

  double getPfIso(const pat::Photon& photon,
                  int pfId,
                  double r1 = 0.4,
                  double r2 = 0.00,
                  double threshold = 0,
                  double jWidth = 0.0,
                  int footprintRemoval = 0);
  double getPfIsoSubUE(const pat::Photon& photon,
                       int pfId,
                       double r1 = 0.4,
                       double r2 = 0.00,
                       double threshold = 0,
                       double jWidth = 0.0,
                       int footprintRemoval = 0,
                       bool excludeCone = false);

  double getPfIso(const reco::GsfElectron& ele, int pfId, double r1 = 0.4, double r2 = 0.00, double threshold = 0);

  enum footprintOptions {
    noRemoval = 0,
    removePFcand,    // remove PF candidates in the isolation map
    removeSCenergy,  // remove SC raw transverse energy
  };

private:
  bool usePackedCandidates_;
  //edm::Handle<edm::View< Tcand >> candidatesView;
  edm::Handle<edm::View<pat::PackedCandidate>> candidatesView;
  reco::Vertex::Point vtx_;
};

#endif
