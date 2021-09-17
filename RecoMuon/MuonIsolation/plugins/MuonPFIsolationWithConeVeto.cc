#include "PhysicsTools/IsolationAlgos/interface/CITKIsolationConeDefinitionBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Math/interface/deltaR.h"

namespace pat {
  typedef edm::Ptr<pat::PackedCandidate> PackedCandidatePtr;
}

class MuonPFIsolationWithConeVeto : public citk::IsolationConeDefinitionBase {
public:
  MuonPFIsolationWithConeVeto(const edm::ParameterSet& c)
      : citk::IsolationConeDefinitionBase(c),
        _vetoThreshold(c.getParameter<double>("VetoThreshold")),
        _vetoConeSize2(std::pow(c.getParameter<double>("VetoConeSize"), 2.0)),
        _miniAODVertexCodes(c.getParameter<std::vector<unsigned> >("miniAODVertexCodes")) {
    char buf[50];
    snprintf(buf, 49, "ThresholdVeto%03.0f-ConeVeto%03.0f", 100 * _vetoThreshold, 100 * std::sqrt(_vetoConeSize2));
    _additionalCode = std::string(buf);
  }
  MuonPFIsolationWithConeVeto(const MuonPFIsolationWithConeVeto&) = delete;
  MuonPFIsolationWithConeVeto& operator=(const MuonPFIsolationWithConeVeto&) = delete;

  void setConsumes(edm::ConsumesCollector) override {}

  bool isInIsolationCone(const reco::CandidatePtr& physob, const reco::CandidatePtr& other) const final;

  //! Destructor
  ~MuonPFIsolationWithConeVeto() override{};

private:
  const double _vetoThreshold, _vetoConeSize2;
  const std::vector<unsigned> _miniAODVertexCodes;
  edm::EDGetTokenT<reco::VertexCollection> _vtxToken;
};

DEFINE_EDM_PLUGIN(CITKIsolationConeDefinitionFactory, MuonPFIsolationWithConeVeto, "MuonPFIsolationWithConeVeto");

bool MuonPFIsolationWithConeVeto::isInIsolationCone(const reco::CandidatePtr& physob,
                                                    const reco::CandidatePtr& iso_obj) const {
  if (iso_obj->pt() <= _vetoThreshold)
    return false;
  const double deltar2 = reco::deltaR2(*physob, *iso_obj);
  if (deltar2 <= _vetoConeSize2 || deltar2 >= _coneSize2)
    return false;

  //the rest will check the vertex selection
  const pat::PackedCandidatePtr aspacked(iso_obj);
  const reco::PFCandidatePtr aspf(iso_obj);

  bool result = true;
  if (aspacked.isNonnull() && aspacked.get()) {
    if (aspacked->charge() != 0) {
      bool is_vertex_allowed = false;
      for (const unsigned vtxtype : _miniAODVertexCodes) {
        if (vtxtype == aspacked->fromPV()) {
          is_vertex_allowed = true;
          break;
        }
      }
      result = result && (is_vertex_allowed);
    }
  } else if (aspf.isNonnull() && aspf.get()) {
  } else {
    throw cms::Exception("InvalidIsolationInput") << "The supplied candidate to be used as isolation "
                                                  << "was neither a reco::PFCandidate nor a pat::PackedCandidate!";
  }
  return result;
}
