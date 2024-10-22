// -*- C++ -*-
//
// Package:    RecoHI/HiJetAlgos/plugins/HiPuRhoProducer
// Class:      HiPuRhoProducer
//
/**\class HiPuRhoProducer HiPuRhoProducer.cc RecoHI/HiJetAlgos/plugins/HiPuRhoProducer.cc
 Description: Producer to dump Pu-jet style rho into event content
 Implementation:
 Just see MultipleAlgoIterator - re-implenting for use in CS jet with sigma subtraction and zeroing
*/
//
// Original Author:  Chris McGinn - in Style of HiFJRhoProducer
//         Created:  Mon, 29 May 2017
//
//
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoHI/HiJetAlgos/plugins/HITowerHelper.h"

#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <map>
#include <vector>

struct EtaPhiTower {
  int ieta, iphi;
  float eta, phi;
};

class HiPuRhoProducer : public edm::stream::EDProducer<> {
public:
  explicit HiPuRhoProducer(const edm::ParameterSet&);

  using ClusterSequencePtr = std::shared_ptr<fastjet::ClusterSequence>;
  using JetDefPtr = std::shared_ptr<fastjet::JetDefinition>;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  virtual void setupGeometryMap(edm::Event& iEvent, const edm::EventSetup& iSetup);
  virtual void calculatePedestal(std::vector<fastjet::PseudoJet> const& coll);
  virtual void subtractPedestal(std::vector<fastjet::PseudoJet>& coll);
  virtual void calculateOrphanInput(std::vector<fastjet::PseudoJet>& orphanInput);
  virtual void putRho(edm::Event& iEvent, const edm::EventSetup& iSetup);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // This checks if the tower is anomalous (if a calo tower).
  virtual void inputTowers();

  bool postOrphan_ = false;
  bool setInitialValue_;

  const bool dropZeroTowers_;
  const int medianWindowWidth_;
  const double minimumTowersFraction_;
  const double nSigmaPU_;  // number of sigma for pileup
  const double puPtMin_;
  const double radiusPU_;  // pileup radius
  const double rParam_;    // the R parameter to use
  const double towSigmaCut_;
  const edm::EDGetTokenT<CaloTowerCollection> caloTowerToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
  const int initialValue_ = -99;
  constexpr static int nEtaTow_ = 82;

  std::array<int, nEtaTow_> vngeom_;
  std::array<int, nEtaTow_> vntow_;
  std::array<float, nEtaTow_> vmean0_;
  std::array<float, nEtaTow_> vrms0_;
  std::array<float, nEtaTow_> vrho0_;
  std::array<float, nEtaTow_> vmean1_;
  std::array<float, nEtaTow_> vrms1_;
  std::array<float, nEtaTow_> vrho1_;

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

  std::vector<const CaloTower*> inputs_;
  ClusterSequencePtr fjClusterSeq_;  // fastjet cluster sequence
  JetDefPtr fjJetDefinition_;        // fastjet jet definition

  std::vector<fastjet::PseudoJet> fjInputs_;          // fastjet inputs
  std::vector<fastjet::PseudoJet> fjJets_;            // fastjet jets
  std::vector<fastjet::PseudoJet> fjOriginalInputs_;  // to back-up unsubtracted fastjet inputs

  CaloGeometry const* geo_ = nullptr;  // geometry

  int ietamax_;                                 // maximum eta in geometry
  int ietamin_;                                 // minimum eta in geometry
  std::map<int, int> ntowersWithJets_;          // number of towers with jets
  std::map<int, int> geomtowers_;               // map of geometry towers to det id
  std::map<int, double> esigma_;                // energy sigma
  std::map<int, double> emean_;                 // energy mean
  std::map<int, std::array<double, 4>> eTop4_;  // energy mean

  typedef std::pair<double, double> EtaPhi;
  std::vector<EtaPhiTower> towermap_;
};

HiPuRhoProducer::HiPuRhoProducer(const edm::ParameterSet& iConfig)
    : dropZeroTowers_(iConfig.getParameter<bool>("dropZeroTowers")),
      medianWindowWidth_(iConfig.getParameter<int>("medianWindowWidth")),
      minimumTowersFraction_(iConfig.getParameter<double>("minimumTowersFraction")),
      nSigmaPU_(iConfig.getParameter<double>("nSigmaPU")),
      puPtMin_(iConfig.getParameter<double>("puPtMin")),
      radiusPU_(iConfig.getParameter<double>("radiusPU")),
      rParam_(iConfig.getParameter<double>("rParam")),
      towSigmaCut_(iConfig.getParameter<double>("towSigmaCut")),
      caloTowerToken_(consumes<CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      caloGeometryToken_(esConsumes<CaloGeometry, CaloGeometryRecord>(edm::ESInputTag{})) {
  //register your products
  produces<std::vector<double>>("mapEtaEdges");
  produces<std::vector<double>>("mapToRho");
  produces<std::vector<double>>("mapToRhoMedian");
  produces<std::vector<double>>("mapToRhoExtra");
  produces<std::vector<double>>("mapToRhoM");
  produces<std::vector<int>>("mapToNTow");
  produces<std::vector<double>>("mapToTowExcludePt");
  produces<std::vector<double>>("mapToTowExcludePhi");
  produces<std::vector<double>>("mapToTowExcludeEta");
}

// ------------ method called to produce the data  ------------
void HiPuRhoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  setupGeometryMap(iEvent, iSetup);

  for (int i = ietamin_; i < ietamax_ + 1; i++) {
    ntowersWithJets_[i] = 0;
  }

  auto const& inputView = iEvent.get(caloTowerToken_);
  inputs_.reserve(inputView.size());
  for (auto const& input : inputView)
    inputs_.push_back(&input);

  fjInputs_.reserve(inputs_.size());
  inputTowers();
  fjOriginalInputs_ = fjInputs_;
  setInitialValue_ = true;
  calculatePedestal(fjInputs_);
  subtractPedestal(fjInputs_);

  fjJetDefinition_ = std::make_shared<fastjet::JetDefinition>(fastjet::antikt_algorithm, rParam_);
  fjClusterSeq_ = std::make_shared<fastjet::ClusterSequence>(fjInputs_, *fjJetDefinition_);
  fjJets_ = fastjet::sorted_by_pt(fjClusterSeq_->inclusive_jets(puPtMin_));

  etaEdgeLow_.clear();
  etaEdgeHi_.clear();
  etaEdges_.clear();

  rho_.clear();
  rhoExtra_.clear();
  rhoM_.clear();
  nTow_.clear();

  towExcludePt_.clear();
  towExcludePhi_.clear();
  towExcludeEta_.clear();

  setInitialValue_ = false;
  std::vector<fastjet::PseudoJet> orphanInput;
  calculateOrphanInput(orphanInput);
  calculatePedestal(orphanInput);
  putRho(iEvent, iSetup);

  inputs_.clear();
  fjInputs_.clear();
  fjJets_.clear();
}

void HiPuRhoProducer::inputTowers() {
  int index = -1;
  for (auto const& input : inputs_) {
    index++;

    if (edm::isNotFinite(input->pt()))
      continue;
    if (input->pt() < 100 * std::numeric_limits<double>::epsilon())
      continue;

    fjInputs_.push_back(fastjet::PseudoJet(input->px(), input->py(), input->pz(), input->energy()));
    fjInputs_.back().set_user_index(index);
  }
}

void HiPuRhoProducer::setupGeometryMap(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  LogDebug("PileUpSubtractor") << "The subtractor setting up geometry...\n";
  const auto& pG = iSetup.getData(caloGeometryToken_);
  geo_ = &pG;
  std::vector<DetId> alldid = geo_->getValidDetIds();
  int ietaold = -10000;
  ietamax_ = -10000;
  ietamin_ = 10000;
  towermap_.clear();

  for (auto const& did : alldid) {
    if (did.det() == DetId::Hcal) {
      HcalDetId hid = HcalDetId(did);
      towermap_.push_back({hid.ieta(), hid.iphi(), geo_->getPosition(did).eta(), geo_->getPosition(did).phi()});
      if (hid.ieta() != ietaold) {
        ietaold = hid.ieta();
        geomtowers_[hid.ieta()] = 1;
        if (hid.ieta() > ietamax_)
          ietamax_ = hid.ieta();
        if (hid.ieta() < ietamin_)
          ietamin_ = hid.ieta();
      } else {
        geomtowers_[hid.ieta()]++;
      }
    }
  }

  for (auto const& gt : geomtowers_) {
    int it = gt.first;
    int vi = it - 1;

    if (it < 0)
      vi = nEtaTow_ + it;

    if (vi < nEtaTow_)
      vngeom_[vi] = gt.second;
  }
}

void HiPuRhoProducer::calculatePedestal(std::vector<fastjet::PseudoJet> const& coll) {
  LogDebug("PileUpSubtractor") << "The subtractor calculating pedestals...\n";

  std::map<int, double> emean2;
  std::map<int, int> ntowers;

  int ietaold = -10000;
  // Initial values for emean_, emean2, esigma_, ntowers

  for (int vi = 0; vi < nEtaTow_; ++vi) {
    vntow_[vi] = initialValue_;
    vmean1_[vi] = initialValue_;
    vrms1_[vi] = initialValue_;
    vrho1_[vi] = initialValue_;

    if (setInitialValue_) {
      vmean0_[vi] = initialValue_;
      vrms0_[vi] = initialValue_;
      vrho0_[vi] = initialValue_;
    }
  }

  for (int i = ietamin_; i < ietamax_ + 1; i++) {
    emean_[i] = 0.;
    emean2[i] = 0.;
    esigma_[i] = 0.;
    ntowers[i] = 0;

    eTop4_[i] = {{0., 0., 0., 0.}};
  }

  for (auto const& input_object : coll) {
    const CaloTower* originalTower = inputs_[input_object.user_index()];
    double original_Et = originalTower->et();
    int ieta0 = originalTower->ieta();

    if (original_Et > eTop4_[ieta0][0]) {
      eTop4_[ieta0][3] = eTop4_[ieta0][2];
      eTop4_[ieta0][2] = eTop4_[ieta0][1];
      eTop4_[ieta0][1] = eTop4_[ieta0][0];
      eTop4_[ieta0][0] = original_Et;
    } else if (original_Et > eTop4_[ieta0][1]) {
      eTop4_[ieta0][3] = eTop4_[ieta0][2];
      eTop4_[ieta0][2] = eTop4_[ieta0][1];
      eTop4_[ieta0][1] = original_Et;
    } else if (original_Et > eTop4_[ieta0][2]) {
      eTop4_[ieta0][3] = eTop4_[ieta0][2];
      eTop4_[ieta0][2] = original_Et;
    } else if (original_Et > eTop4_[ieta0][3]) {
      eTop4_[ieta0][3] = original_Et;
    }

    emean_[ieta0] = emean_[ieta0] + original_Et;
    emean2[ieta0] = emean2[ieta0] + original_Et * original_Et;
    if (ieta0 - ietaold != 0) {
      ntowers[ieta0] = 1;
      ietaold = ieta0;
    } else {
      ntowers[ieta0]++;
    }
  }

  for (auto const& gt : geomtowers_) {
    int it = gt.first;

    int vi = it - 1;

    if (it < 0)
      vi = nEtaTow_ + it;

    double e1 = emean_[it];
    double e2 = emean2[it];
    int nt = gt.second - ntowersWithJets_[it];

    if (vi < nEtaTow_) {
      vntow_[vi] = nt;
    }

    LogDebug("PileUpSubtractor") << " ieta: " << it << " number of towers: " << nt << " e1: " << e1 << " e2: " << e2
                                 << "\n";

    if (nt > 0) {
      if (postOrphan_) {
        if (nt > (int)minimumTowersFraction_ * (gt.second)) {
          emean_[it] = e1 / (double)nt;
          double eee = e2 / (double)nt - e1 * e1 / (double)(nt * nt);
          if (eee < 0.)
            eee = 0.;
          esigma_[it] = nSigmaPU_ * sqrt(eee);

          uint32_t numToCheck = std::min(int(eTop4_[it].size()), nt - (int)minimumTowersFraction_ * (gt.second));

          for (unsigned int lI = 0; lI < numToCheck; ++lI) {
            if (eTop4_[it][lI] >= emean_[it] + towSigmaCut_ * esigma_[it] && towSigmaCut_ > 0) {
              e1 -= eTop4_[it][lI];
              nt -= 1;
            } else
              break;
          }

          if (e1 < .000000001)
            e1 = 0;
        }
      }

      emean_[it] = e1 / (double)nt;
      double eee = e2 / nt - e1 * e1 / (nt * nt);
      if (eee < 0.)
        eee = 0.;
      esigma_[it] = nSigmaPU_ * sqrt(eee);

      double etaWidth = std::abs(hi::etaedge[abs(it)] - hi::etaedge[abs(it) - 1]);

      int sign = (it < 0) ? -1 : 1;
      auto absIt = std::abs(it);
      if (it < 0) {
        etaEdgeLow_.push_back(sign * hi::etaedge[absIt]);
        etaEdgeHi_.push_back(sign * hi::etaedge[absIt - 1]);
      } else {
        etaEdgeHi_.push_back(sign * hi::etaedge[absIt]);
        etaEdgeLow_.push_back(sign * hi::etaedge[absIt - 1]);
      }

      if (vi < nEtaTow_) {
        vmean1_[vi] = emean_[it];
        vrho1_[vi] = emean_[it] / (etaWidth * (2. * M_PI / (double)vngeom_[vi]));
        rho_.push_back(vrho1_[vi]);
        rhoM_.push_back(0);
        vrms1_[vi] = esigma_[it];
        if (vngeom_[vi] == vntow_[vi]) {
          vmean0_[vi] = emean_[it];
          vrho0_[vi] = emean_[it] / (etaWidth * (2. * M_PI / (double)vngeom_[vi]));
          vrms0_[vi] = esigma_[it];
        }
        rhoExtra_.push_back(vrho0_[vi]);
        nTow_.push_back(vntow_[vi]);
      }
    } else {
      emean_[it] = 0.;
      esigma_[it] = 0.;
    }
    LogDebug("PileUpSubtractor") << " ieta: " << it << " Pedestals: " << emean_[it] << "  " << esigma_[it] << "\n";
  }

  postOrphan_ = false;
}

void HiPuRhoProducer::subtractPedestal(std::vector<fastjet::PseudoJet>& coll) {
  LogDebug("PileUpSubtractor") << "The subtractor subtracting pedestals...\n";

  std::vector<fastjet::PseudoJet> newcoll;
  for (auto& input_object : coll) {
    int index = input_object.user_index();
    CaloTower const* itow = inputs_[index];
    int it = itow->ieta();

    double original_Et = itow->et();
    double etnew = original_Et - emean_.at(it) - esigma_.at(it);
    float mScale = etnew / input_object.Et();
    if (etnew < 0.)
      mScale = 0.;

    math::XYZTLorentzVectorD towP4(
        input_object.px() * mScale, input_object.py() * mScale, input_object.pz() * mScale, input_object.e() * mScale);

    input_object.reset_momentum(towP4.px(), towP4.py(), towP4.pz(), towP4.energy());
    input_object.set_user_index(index);

    if (etnew > 0. && dropZeroTowers_)
      newcoll.push_back(input_object);
  }

  if (dropZeroTowers_)
    coll = newcoll;
}

void HiPuRhoProducer::calculateOrphanInput(std::vector<fastjet::PseudoJet>& orphanInput) {
  LogDebug("PileUpSubtractor") << "The subtractor calculating orphan input...\n";

  fjInputs_ = fjOriginalInputs_;

  std::vector<int> jettowers;                         // vector of towers indexed by "user_index"
  std::map<std::pair<int, int>, int> excludedTowers;  // map of excluded ieta, iphi values
  for (auto const& pseudojetTMP : fjJets_) {
    EtaPhi jet_etaphi(pseudojetTMP.eta(), pseudojetTMP.phi());

    if (pseudojetTMP.perp() < puPtMin_)
      continue;

    for (auto const& im : towermap_) {
      double dr2 = reco::deltaR2(im.eta, im.phi, jet_etaphi.first, jet_etaphi.second);
      if (dr2 < radiusPU_ * radiusPU_ && !excludedTowers[std::pair(im.ieta, im.iphi)] &&
          (geomtowers_[im.ieta] - ntowersWithJets_[im.ieta]) > minimumTowersFraction_ * geomtowers_[im.ieta]) {
        ntowersWithJets_[im.ieta]++;
        excludedTowers[std::pair(im.ieta, im.iphi)] = 1;
      }
    }

    for (auto const& input : fjInputs_) {
      int index = input.user_index();
      const CaloTower* originalTower = inputs_[index];
      int ie = originalTower->ieta();
      int ip = originalTower->iphi();

      if (excludedTowers[std::pair<int, int>(ie, ip)]) {
        jettowers.push_back(index);
      }
    }
  }
  // Create a new collections from the towers not included in jets
  for (auto const& input : fjInputs_) {
    int index = input.user_index();
    const CaloTower* originalTower = inputs_[index];
    auto itjet = std::find(jettowers.begin(), jettowers.end(), index);
    if (itjet == jettowers.end()) {
      orphanInput.emplace_back(originalTower->px(), originalTower->py(), originalTower->pz(), originalTower->energy());
      orphanInput.back().set_user_index(index);
    } else {
      towExcludePt_.push_back(originalTower->pt());
      towExcludePhi_.push_back(originalTower->phi());
      towExcludeEta_.push_back(originalTower->eta());
    }
  }

  postOrphan_ = true;
}

void HiPuRhoProducer::putRho(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::size_t size = etaEdgeLow_.size();

  std::vector<std::pair<std::size_t, double>> order;
  for (std::size_t i = 0; i < size; ++i) {
    order.emplace_back(i, etaEdgeLow_[i]);
  }

  std::sort(
      order.begin(), order.end(), [](auto const& pair0, auto const& pair1) { return pair0.second < pair1.second; });

  std::vector<double> sortedEtaEdgeLow(size);
  std::vector<double> sortedEtaEdgeHigh(size);

  auto mapToRhoOut = std::make_unique<std::vector<double>>(size);
  auto mapToRhoExtraOut = std::make_unique<std::vector<double>>(size);
  auto mapToRhoMOut = std::make_unique<std::vector<double>>(size);
  auto mapToNTowOut = std::make_unique<std::vector<int>>(size);

  for (std::size_t i = 0; i < size; ++i) {
    auto const& index = order[i].first;

    sortedEtaEdgeLow[i] = etaEdgeLow_[index];
    sortedEtaEdgeHigh[i] = etaEdgeHi_[index];

    (*mapToRhoOut)[i] = rho_[index];
    (*mapToRhoExtraOut)[i] = rhoExtra_[index];
    (*mapToRhoMOut)[i] = rhoM_[index];
    (*mapToNTowOut)[i] = nTow_[index];
  }

  auto mapToRhoMedianOut = std::make_unique<std::vector<double>>(size);

  for (uint32_t index = medianWindowWidth_; index < size - medianWindowWidth_; ++index) {
    auto centre = mapToRhoOut->begin() + index;
    std::vector<float> rhoRange(centre - medianWindowWidth_, centre + medianWindowWidth_);
    std::nth_element(rhoRange.begin(), rhoRange.begin() + medianWindowWidth_, rhoRange.end());
    (*mapToRhoMedianOut)[index] = rhoRange[medianWindowWidth_];
  }

  auto mapEtaRangesOut = std::make_unique<std::vector<double>>();

  mapEtaRangesOut->assign(sortedEtaEdgeLow.begin(), sortedEtaEdgeLow.end());
  mapEtaRangesOut->push_back(sortedEtaEdgeHigh.back());

  auto mapToTowExcludePtOut = std::make_unique<std::vector<double>>(std::move(towExcludePt_));
  auto mapToTowExcludePhiOut = std::make_unique<std::vector<double>>(std::move(towExcludePhi_));
  auto mapToTowExcludeEtaOut = std::make_unique<std::vector<double>>(std::move(towExcludeEta_));

  iEvent.put(std::move(mapEtaRangesOut), "mapEtaEdges");
  iEvent.put(std::move(mapToRhoOut), "mapToRho");
  iEvent.put(std::move(mapToRhoMedianOut), "mapToRhoMedian");
  iEvent.put(std::move(mapToRhoExtraOut), "mapToRhoExtra");
  iEvent.put(std::move(mapToRhoMOut), "mapToRhoM");
  iEvent.put(std::move(mapToNTowOut), "mapToNTow");
  iEvent.put(std::move(mapToTowExcludePtOut), "mapToTowExcludePt");
  iEvent.put(std::move(mapToTowExcludePhiOut), "mapToTowExcludePhi");
  iEvent.put(std::move(mapToTowExcludeEtaOut), "mapToTowExcludeEta");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HiPuRhoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("PFTowers"));
  desc.add<int>("medianWindowWidth", 2);
  desc.add<double>("towSigmaCut", 5.0);
  desc.add<double>("puPtMin", 15.0);
  desc.add<double>("rParam", 0.3);
  desc.add<double>("nSigmaPU", 1.0);
  desc.add<double>("radiusPU", 0.5);
  desc.add<double>("minimumTowersFraction", 0.7);
  desc.add<bool>("dropZeroTowers", true);
  descriptions.add("hiPuRhoProducer", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiPuRhoProducer);
