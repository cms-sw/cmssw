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

#include "RecoHI/HiJetAlgos/plugins/HiPuRhoProducer.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
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
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "TMath.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HiPuRhoProducer::HiPuRhoProducer(const edm::ParameterSet& iConfig)
    : dropZeroTowers_(iConfig.getParameter<bool>("dropZeroTowers")),
      medianWindowWidth_(iConfig.getParameter<int>("medianWindowWidth")),
      minimumTowersFraction_(iConfig.getParameter<double>("minimumTowersFraction")),
      nSigmaPU_(iConfig.getParameter<double>("nSigmaPU")),
      puPtMin_(iConfig.getParameter<double>("puPtMin")),
      radiusPU_(iConfig.getParameter<double>("radiusPU")),
      rParam_(iConfig.getParameter<double>("rParam")),
      src_(iConfig.getParameter<edm::InputTag>("src")),
      towSigmaCut_(iConfig.getParameter<double>("towSigmaCut")) {
  input_candidateview_token_ = consumes<reco::CandidateView>(src_);

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

HiPuRhoProducer::~HiPuRhoProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called to produce the data  ------------
void HiPuRhoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  setupGeometryMap(iEvent, iSetup);

  inputs_.clear();
  fjInputs_.clear();
  fjJets_.clear();

  // Get the particletowers
  edm::Handle<reco::CandidateView> inputsHandle;
  iEvent.getByToken(input_candidateview_token_, inputsHandle);

  for (size_t i = 0; i < inputsHandle->size(); ++i)
    inputs_.push_back(inputsHandle->ptrAt(i));

  fjInputs_.reserve(inputs_.size());
  inputTowers();
  fjOriginalInputs_ = fjInputs_;

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

  std::vector<fastjet::PseudoJet> orphanInput;
  calculateOrphanInput(orphanInput);
  calculatePedestal(orphanInput);
  putRho(iEvent, iSetup);
}

void HiPuRhoProducer::inputTowers() {
  auto inBegin = inputs_.begin();
  auto inEnd = inputs_.end();
  for (auto i = inBegin; i != inEnd; ++i) {
    reco::CandidatePtr input = *i;

    if (edm::isNotFinite(input->pt()))
      continue;
    if (input->pt() < 100 * std::numeric_limits<double>::epsilon())
      continue;

    fjInputs_.push_back(fastjet::PseudoJet(input->px(), input->py(), input->pz(), input->energy()));
    fjInputs_.back().set_user_index(i - inBegin);
  }
}

void HiPuRhoProducer::setupGeometryMap(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  LogDebug("PileUpSubtractor") << "The subtractor setting up geometry...\n";

  if (geo_ == nullptr) {
    edm::ESHandle<CaloGeometry> pG;
    iSetup.get<CaloGeometryRecord>().get(pG);
    geo_ = pG.product();
    std::vector<DetId> alldid = geo_->getValidDetIds();

    int ietaold = -10000;
    ietamax_ = -10000;
    ietamin_ = 10000;
    for (auto const& did : alldid) {
      if (did.det() == DetId::Hcal) {
        HcalDetId hid = HcalDetId(did);
        allgeomid_.push_back(did);

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
  }

  for (int i = ietamin_; i < ietamax_ + 1; i++) {
    ntowersWithJets_[i] = 0;
  }
}

void HiPuRhoProducer::calculatePedestal(std::vector<fastjet::PseudoJet> const& coll) {
  LogDebug("PileUpSubtractor") << "The subtractor calculating pedestals...\n";

  std::map<int, double> emean2;
  std::map<int, int> ntowers;

  int ietaold = -10000;
  // Initial values for emean_, emean2, esigma_, ntowers

  for (int vi = 0; vi < nEtaTow_; ++vi) {
    int it = vi + 1;
    if (it > nEtaTow_ / 2)
      it = vi - nEtaTow_;

    int sign = (it < 0) ? -1 : 1;

    vieta[vi] = it;
    veta[vi] = sign * (etaedge[abs(it)] + etaedge[abs(it) - 1]) / 2.;
    vngeom[vi] = -99;
    vntow[vi] = -99;

    vmean1[vi] = -99;
    vrms1[vi] = -99;
    vrho1[vi] = -99;

    if ((*(ntowersWithJets_.find(it))).second == 0) {
      vmean0[vi] = -99;
      vrms0[vi] = -99;
      vrho0[vi] = -99;
    }
  }

  for (int i = ietamin_; i < ietamax_ + 1; i++) {
    emean_[i] = 0.;
    emean2[i] = 0.;
    esigma_[i] = 0.;
    ntowers[i] = 0;

    eTop4_[i] = {0., 0., 0., 0.};
  }

  for (auto const& input_object : coll) {
    const reco::CandidatePtr& originalTower = inputs_[input_object.user_index()];
    double Original_Et = originalTower->et();
    int ieta0 = ieta(originalTower);

    if (Original_Et > eTop4_[ieta0][0]) {
      eTop4_[ieta0][3] = eTop4_[ieta0][2];
      eTop4_[ieta0][2] = eTop4_[ieta0][1];
      eTop4_[ieta0][1] = eTop4_[ieta0][0];
      eTop4_[ieta0][0] = Original_Et;
    } else if (Original_Et > eTop4_[ieta0][1]) {
      eTop4_[ieta0][3] = eTop4_[ieta0][2];
      eTop4_[ieta0][2] = eTop4_[ieta0][1];
      eTop4_[ieta0][1] = Original_Et;
    } else if (Original_Et > eTop4_[ieta0][2]) {
      eTop4_[ieta0][3] = eTop4_[ieta0][2];
      eTop4_[ieta0][2] = Original_Et;
    } else if (Original_Et > eTop4_[ieta0][3]) {
      eTop4_[ieta0][3] = Original_Et;
    }

    if (ieta0 - ietaold != 0) {
      emean_[ieta0] = emean_[ieta0] + Original_Et;
      emean2[ieta0] = emean2[ieta0] + Original_Et * Original_Et;
      ntowers[ieta0] = 1;
      ietaold = ieta0;
    } else {
      emean_[ieta0] = emean_[ieta0] + Original_Et;
      emean2[ieta0] = emean2[ieta0] + Original_Et * Original_Et;
      ntowers[ieta0]++;
    }
  }

  for (auto const& gt : geomtowers_) {
    int it = gt.first;

    int vi = it - 1;

    if (it < 0)
      vi = nEtaTow_ + it;

    double e1 = (emean_.find(it))->second;
    double e2 = (emean2.find(it))->second;
    int nt = gt.second - (ntowersWithJets_.find(it))->second;

    if (vi < nEtaTow_) {
      vngeom[vi] = gt.second;
      vntow[vi] = nt;
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

      double etaWidth = etaedge[abs(it)] - etaedge[abs(it) - 1];
      if (etaWidth < 0)
        etaWidth *= -1.;

      int sign = (it < 0) ? -1 : 1;

      if (sign * etaedge[abs(it)] < sign * etaedge[abs(it) - 1]) {
        etaEdgeLow_.push_back(sign * etaedge[abs(it)]);
        etaEdgeHi_.push_back(sign * etaedge[abs(it) - 1]);
      } else {
        etaEdgeHi_.push_back(sign * etaedge[abs(it)]);
        etaEdgeLow_.push_back(sign * etaedge[abs(it) - 1]);
      }

      if (vi < nEtaTow_) {
        vmean1[vi] = emean_[it];
        vrho1[vi] = emean_[it] / (etaWidth * (2. * TMath::Pi() / (double)vngeom[vi]));
        rho_.push_back(vrho1[vi]);
        rhoM_.push_back(0);
        vrms1[vi] = esigma_[it];
        if (vngeom[vi] == vntow[vi]) {
          vmean0[vi] = emean_[it];
          vrho0[vi] = emean_[it] / (etaWidth * (2. * TMath::Pi() / (double)vngeom[vi]));
          vrms0[vi] = esigma_[it];
        }
        rhoExtra_.push_back(vrho0[vi]);
        nTow_.push_back(vntow[vi]);
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
    reco::CandidatePtr const& itow = inputs_[index];

    int it = ieta(itow);
    iphi(itow);

    double Original_Et = itow->et();
    double etnew = Original_Et - (emean_.find(it))->second - (esigma_.find(it))->second;
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

  std::vector<int> jettowers;                       // vector of towers indexed by "user_index"
  std::vector<std::pair<int, int>> excludedTowers;  // vector of excluded ieta, iphi values

  int32_t nref = 0;

  for (auto const& pseudojetTMP : fjJets_) {
    if (nref < nMaxJets_) {
      jtexngeom[nref] = 0;
      jtexntow[nref] = 0;
      jtexpt[nref] = 0;
      jtpt[nref] = pseudojetTMP.perp();
      jteta[nref] = pseudojetTMP.eta();
      jtphi[nref] = pseudojetTMP.phi();
    }

    if (pseudojetTMP.perp() < puPtMin_)
      continue;

    for (auto const& im : allgeomid_) {
      double dr = reco::deltaR(geo_->getPosition((DetId)im), pseudojetTMP);
      std::vector<std::pair<int, int>>::const_iterator exclude =
          std::find(excludedTowers.begin(), excludedTowers.end(), std::pair<int, int>(im.ieta(), im.iphi()));
      if (dr < radiusPU_ && exclude == excludedTowers.end() &&
          (geomtowers_[im.ieta()] - ntowersWithJets_[im.ieta()]) > minimumTowersFraction_ * (geomtowers_[im.ieta()])) {
        ntowersWithJets_[im.ieta()]++;
        excludedTowers.emplace_back(std::pair<int, int>(im.ieta(), im.iphi()));

        if (nref < nMaxJets_)
          jtexngeom[nref]++;
      }
    }

    for (auto const& input : fjInputs_) {
      int index = input.user_index();
      const reco::CandidatePtr& originalTower = inputs_[index];

      int ie = ieta(originalTower);
      int ip = iphi(originalTower);
      auto exclude = std::find(excludedTowers.begin(), excludedTowers.end(), std::pair<int, int>(ie, ip));
      if (exclude != excludedTowers.end()) {
        jettowers.push_back(index);
      }

      double dr = reco::deltaR(input, pseudojetTMP);
      if (dr < radiusPU_ && nref < nMaxJets_) {
        jtexntow[nref]++;
        jtexpt[nref] += originalTower->pt();
      }
    }

    if (nref < nMaxJets_)
      nref++;
  }

  // Create a new collections from the towers not included in jets
  for (auto const& input : fjInputs_) {
    int index = input.user_index();
    const reco::CandidatePtr& originalTower = inputs_[index];
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
    order.emplace_back(std::make_pair(i, etaEdgeLow_[i]));
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

// ------------ method called once each job just before starting event loop  ------------
void HiPuRhoProducer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void HiPuRhoProducer::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HiPuRhoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

int HiPuRhoProducer::ieta(const reco::CandidatePtr& in) const {
  const CaloTower* ctc = dynamic_cast<const CaloTower*>(in.get());
  if (ctc) {
    return ctc->id().ieta();
  }

  throw cms::Exception("Invalid Constituent") << "CaloJet constituent is not of CaloTower type";

  return 0;
}

int HiPuRhoProducer::iphi(const reco::CandidatePtr& in) const {
  const CaloTower* ctc = dynamic_cast<const CaloTower*>(in.get());
  if (ctc) {
    return ctc->id().iphi();
  }

  throw cms::Exception("Invalid Constituent") << "CaloJet constituent is not of CaloTower type";

  return 0;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiPuRhoProducer);
