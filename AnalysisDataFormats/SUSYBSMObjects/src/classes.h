#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"
namespace {
 namespace {
  susybsm::HSCParticle pa;
  susybsm::DriftTubeTOF dtitof;

  susybsm::TimeMeasurement tm;
  std::vector<susybsm::TimeMeasurement> tmv;
  susybsm::MuonTOFCollection mtc; 
  susybsm::MuonTOF mt;
  susybsm::MuonTOFRef mtr;
  susybsm::MuonTOFRefProd mtrp;
  susybsm::MuonTOFRefVector mtrv;
  edm::Wrapper<susybsm::MuonTOFCollection> wr;
  std::vector<std::pair<edm::Ref<std::vector<reco::Muon>,reco::Muon,edm::refhelper::FindUsingAdvance<std::vector<reco::Muon>,reco::Muon> >,susybsm::DriftTubeTOF> > a;
  std::vector<susybsm::DriftTubeTOF> b;

  susybsm::DeDxBeta dedxbeta;
  susybsm::HSCParticleCollection hc;
  susybsm::HSCParticleRef hr;
  susybsm::HSCParticleRefProd hp;
  susybsm::HSCParticleRefVector hv;
  edm::Wrapper<susybsm::HSCParticleCollection> wr1;
  
 }
}
