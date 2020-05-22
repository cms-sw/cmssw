#ifndef RecoHI_HiJetAlgos_HiPuRhoProducer_h
#define RecoHI_HiJetAlgos_HiPuRhoProducer_h

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"

#include <map>
#include <vector>

//
// class declaration
//

class HiPuRhoProducer : public edm::EDProducer {
public:
  explicit HiPuRhoProducer(const edm::ParameterSet&);
  ~HiPuRhoProducer() override;

  using ClusterSequencePtr = std::shared_ptr<fastjet::ClusterSequence>;
  using JetDefPtr = std::shared_ptr<fastjet::JetDefinition>;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  virtual void setupGeometryMap(edm::Event& iEvent, const edm::EventSetup& iSetup);
  virtual void calculatePedestal(std::vector<fastjet::PseudoJet> const& coll);
  virtual void subtractPedestal(std::vector<fastjet::PseudoJet>& coll);
  virtual void calculateOrphanInput(std::vector<fastjet::PseudoJet>& orphanInput);
  virtual void putRho(edm::Event& iEvent, const edm::EventSetup& iSetup);

  constexpr static int nMaxJets_ = 200;

  float jteta[nMaxJets_];
  float jtphi[nMaxJets_];
  float jtpt[nMaxJets_];
  float jtpu[nMaxJets_];
  float jtexpt[nMaxJets_];
  int jtexngeom[nMaxJets_];
  int jtexntow[nMaxJets_];

  constexpr static int nEtaTow_ = 82;

  int vngeom[nEtaTow_];
  int vntow[nEtaTow_];
  int vieta[nEtaTow_];
  float veta[nEtaTow_];
  float vmean0[nEtaTow_];
  float vrms0[nEtaTow_];
  float vrho0[nEtaTow_];
  float vmean1[nEtaTow_];
  float vrms1[nEtaTow_];
  float vrho1[nEtaTow_];

  const double etaedge[42] = {0.000, 0.087, 0.174, 0.261, 0.348, 0.435, 0.522, 0.609, 0.696, 0.783, 0.870,
                              0.957, 1.044, 1.131, 1.218, 1.305, 1.392, 1.479, 1.566, 1.653, 1.740, 1.830,
                              1.930, 2.043, 2.172, 2.322, 2.500, 2.650, 2.853, 3.000, 3.139, 3.314, 3.489,
                              3.664, 3.839, 4.013, 4.191, 4.363, 4.538, 4.716, 4.889, 5.191};

  std::vector<double> etaEdgeLow_;
  std::vector<double> etaEdgeHi_;
  std::vector<double> etaEdges_;

  std::vector<double> rho_;
  std::vector<double> rhoExtra_;
  std::vector<double> rhoM_;
  std::vector<int> nTow_;

  std::vector<double> towExcludePt_;
  std::vector<double> towExcludePhi_;
  std::vector<double> towExcludeEta_;

  int ieta(const reco::CandidatePtr& in) const;
  int iphi(const reco::CandidatePtr& in) const;

private:
  void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // This checks if the tower is anomalous (if a calo tower).
  virtual void inputTowers();

  bool postOrphan_;

  bool dropZeroTowers_;
  int medianWindowWidth_;
  double minimumTowersFraction_;
  double nSigmaPU_;                                   // number of sigma for pileup
  double puPtMin_;
  double radiusPU_;                                   // pileup radius
  double rParam_;                                     // the R parameter to use
  edm::InputTag src_;                                 // input constituent source
  double towSigmaCut_;

  std::vector<edm::Ptr<reco::Candidate>> inputs_;     // input candidates
  ClusterSequencePtr fjClusterSeq_;                   // fastjet cluster sequence
  JetDefPtr fjJetDefinition_;                         // fastjet jet definition

  std::vector<fastjet::PseudoJet> fjInputs_;          // fastjet inputs
  std::vector<fastjet::PseudoJet> fjJets_;            // fastjet jets
  std::vector<fastjet::PseudoJet> fjOriginalInputs_;  // to back-up unsubtracted fastjet inputs

  CaloGeometry const* geo_ = nullptr;  // geometry
  std::vector<HcalDetId> allgeomid_;   // all det ids in the geometry

  int ietamax_;                                // maximum eta in geometry
  int ietamin_;                                // minimum eta in geometry
  std::map<int, int> ntowersWithJets_;         // number of towers with jets
  std::map<int, int> geomtowers_;              // map of geometry towers to det id
  std::map<int, double> esigma_;               // energy sigma
  std::map<int, double> emean_;                // energy mean
  std::map<int, std::vector<double>> eTop4_;   // energy mean

  edm::EDGetTokenT<reco::CandidateView> input_candidateview_token_;
};

#endif
