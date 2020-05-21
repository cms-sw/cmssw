#ifndef ParticleTowerProducer_h
#define ParticleTowerProducer_h

// user include files
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include <map>
#include <utility>

class ParticleTowerProducer : public edm::EDProducer {
public:
  explicit ParticleTowerProducer(const edm::ParameterSet&);
  ~ParticleTowerProducer() override;

private:
  void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  void resetTowers(edm::Event& iEvent, const edm::EventSetup& iSetup);
  DetId getNearestTower(const reco::PFCandidate& in) const;
  DetId getNearestTower(double eta, double phi) const;
  //  uint32_t denseIndex(int ieta, int iphi, double eta) const;
  int eta2ieta(double eta) const;
  int phi2iphi(double phi, int ieta) const;
  double ieta2eta(int ieta) const;
  double iphi2phi(int iphi, int ieta) const;
  // ----------member data ---------------------------

  edm::EDGetTokenT<reco::PFCandidateCollection> src_;
  bool useHF_;

  typedef std::pair<int, int> EtaPhi;
  typedef std::map<EtaPhi, double> EtaPhiMap;
  EtaPhiMap towers_;

  double PI;

  CaloGeometry const* geo_;  // geometry

  // tower edges from fast sim, used starting at index 30 for the HF
  const double etaedge[42] = {0.000, 0.087, 0.174, 0.261, 0.348, 0.435, 0.522, 0.609, 0.696, 0.783, 0.870,
                              0.957, 1.044, 1.131, 1.218, 1.305, 1.392, 1.479, 1.566, 1.653, 1.740, 1.830,
                              1.930, 2.043, 2.172, 2.322, 2.500, 2.650, 2.853, 3.000, 3.139, 3.314, 3.489,
                              3.664, 3.839, 4.013, 4.191, 4.363, 4.538, 4.716, 4.889, 5.191};
};
#endif
