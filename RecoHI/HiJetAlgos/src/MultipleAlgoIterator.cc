#include <memory>

#include "RecoHI/HiJetAlgos/interface/MultipleAlgoIterator.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
using namespace std;

MultipleAlgoIterator::MultipleAlgoIterator(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC)
    : PileUpSubtractor(iConfig, std::move(iC)),
      minimumTowersFraction_(iConfig.getParameter<double>("minimumTowersFraction")),
      sumRecHits_(iConfig.getParameter<bool>("sumRecHits")),
      dropZeroTowers_(iConfig.getUntrackedParameter<bool>("dropZeroTowers", true)) {
  LogDebug("PileUpSubtractor") << "LIMITING THE MINIMUM TOWERS FRACTION TO : " << minimumTowersFraction_ << endl;
}

void MultipleAlgoIterator::rescaleRMS(double s) {
  for (std::map<int, double>::iterator iter = esigma_.begin(); iter != esigma_.end(); ++iter) {
    iter->second = s * (iter->second);
  }
}

void MultipleAlgoIterator::offsetCorrectJets() {
  LogDebug("PileUpSubtractor") << "The subtractor correcting jets...\n";
  jetOffset_.clear();

  using namespace reco;

  (*fjInputs_) = fjOriginalInputs_;
  rescaleRMS(nSigmaPU_);
  subtractPedestal(*fjInputs_);
  const fastjet::JetDefinition& def = *fjJetDefinition_;
  if (!doAreaFastjet_ && !doRhoFastjet_) {
    fjClusterSeq_ = std::make_shared<fastjet::ClusterSequence>(*fjInputs_, def);
  } else {
    fjClusterSeq_ = ClusterSequencePtr(new fastjet::ClusterSequenceArea(*fjInputs_, def, *fjActiveArea_));
  }

  (*fjJets_) = fastjet::sorted_by_pt(fjClusterSeq_->inclusive_jets(jetPtMin_));

  jetOffset_.reserve(fjJets_->size());

  vector<fastjet::PseudoJet>::iterator pseudojetTMP = fjJets_->begin(), jetsEnd = fjJets_->end();
  for (; pseudojetTMP != jetsEnd; ++pseudojetTMP) {
    int ijet = pseudojetTMP - fjJets_->begin();
    jetOffset_[ijet] = 0;

    std::vector<fastjet::PseudoJet> towers = sorted_by_pt(pseudojetTMP->constituents());

    double newjetet = 0.;
    for (vector<fastjet::PseudoJet>::const_iterator ito = towers.begin(), towEnd = towers.end(); ito != towEnd; ++ito) {
      const reco::CandidatePtr& originalTower = (*inputs_)[ito->user_index()];
      int it = ieta(originalTower);
      double Original_Et = originalTower->et();
      double etnew = Original_Et - (*emean_.find(it)).second - (*esigma_.find(it)).second;
      if (etnew < 0.)
        etnew = 0;
      newjetet = newjetet + etnew;
      jetOffset_[ijet] += Original_Et - etnew;
    }
  }
}

void MultipleAlgoIterator::subtractPedestal(vector<fastjet::PseudoJet>& coll) {
  LogDebug("PileUpSubtractor") << "The subtractor subtracting pedestals...\n";

  int it = -100;

  vector<fastjet::PseudoJet> newcoll;

  for (vector<fastjet::PseudoJet>::iterator input_object = coll.begin(), fjInputsEnd = coll.end();
       input_object != fjInputsEnd;
       ++input_object) {
    reco::CandidatePtr const& itow = (*inputs_)[input_object->user_index()];

    it = ieta(itow);
    iphi(itow);

    double Original_Et = itow->et();
    if (sumRecHits_) {
      Original_Et = getEt(itow);
    }

    double etnew = Original_Et - (*(emean_.find(it))).second - (*(esigma_.find(it))).second;
    float mScale = etnew / input_object->Et();
    if (etnew < 0.)
      mScale = 0.;

    math::XYZTLorentzVectorD towP4(input_object->px() * mScale,
                                   input_object->py() * mScale,
                                   input_object->pz() * mScale,
                                   input_object->e() * mScale);

    int index = input_object->user_index();
    input_object->reset_momentum(towP4.px(), towP4.py(), towP4.pz(), towP4.energy());
    input_object->set_user_index(index);

    if (etnew > 0. && dropZeroTowers_)
      newcoll.push_back(*input_object);
  }

  if (dropZeroTowers_)
    coll = newcoll;
}

void MultipleAlgoIterator::calculatePedestal(vector<fastjet::PseudoJet> const& coll) {
  LogDebug("PileUpSubtractor") << "The subtractor calculating pedestals...\n";

  map<int, double> emean2;
  map<int, int> ntowers;

  int ietaold = -10000;
  int ieta0 = -100;

  // Initial values for emean_, emean2, esigma_, ntowers

  for (int i = ietamin_; i < ietamax_ + 1; i++) {
    emean_[i] = 0.;
    emean2[i] = 0.;
    esigma_[i] = 0.;
    ntowers[i] = 0;
  }

  for (vector<fastjet::PseudoJet>::const_iterator input_object = coll.begin(), fjInputsEnd = coll.end();
       input_object != fjInputsEnd;
       ++input_object) {
    const reco::CandidatePtr& originalTower = (*inputs_)[input_object->user_index()];
    ieta0 = ieta(originalTower);
    double Original_Et = originalTower->et();
    if (sumRecHits_) {
      Original_Et = getEt(originalTower);
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

  for (map<int, int>::const_iterator gt = geomtowers_.begin(); gt != geomtowers_.end(); gt++) {
    int it = (*gt).first;

    double e1 = (*(emean_.find(it))).second;
    double e2 = (*emean2.find(it)).second;
    int nt = (*gt).second - (*(ntowersWithJets_.find(it))).second;

    LogDebug("PileUpSubtractor") << " ieta : " << it << " number of towers : " << nt << " e1 : " << e1 << " e2 : " << e2
                                 << "\n";

    if (nt > 0) {
      emean_[it] = e1 / nt;
      double eee = e2 / nt - e1 * e1 / (nt * nt);
      if (eee < 0.)
        eee = 0.;
      esigma_[it] = nSigmaPU_ * sqrt(eee);
    } else {
      emean_[it] = 0.;
      esigma_[it] = 0.;
    }
    LogDebug("PileUpSubtractor") << " ieta : " << it << " Pedestals : " << emean_[it] << "  " << esigma_[it] << "\n";
  }
}

void MultipleAlgoIterator::calculateOrphanInput(vector<fastjet::PseudoJet>& orphanInput) {
  LogDebug("PileUpSubtractor") << "The subtractor calculating orphan input...\n";

  (*fjInputs_) = fjOriginalInputs_;

  vector<int> jettowers;                   // vector of towers indexed by "user_index"
  vector<pair<int, int> > excludedTowers;  // vector of excluded ieta, iphi values

  for (auto const& pseudojetTMP : *fjJets_) {
    if (pseudojetTMP.perp() < puPtMin_)
      continue;

    // find towers within radiusPU_ of this jet
    for (auto const im : allgeomid_) {
      double dr = reco::deltaR(geo_->getPosition(im), pseudojetTMP);
      auto exclude = find(excludedTowers.begin(), excludedTowers.end(), pair<int, int>(im.ieta(), im.iphi()));
      if (dr < radiusPU_ && exclude == excludedTowers.end() &&
          (geomtowers_[im.ieta()] - ntowersWithJets_[im.ieta()]) > minimumTowersFraction_ * (geomtowers_[im.ieta()])) {
        ntowersWithJets_[im.ieta()]++;

        excludedTowers.push_back(pair<int, int>(im.ieta(), im.iphi()));
      }
    }

    for (auto const& it : *fjInputs_) {
      int index = it.user_index();
      int ie = ieta((*inputs_)[index]);
      int ip = iphi((*inputs_)[index]);
      auto exclude = find(excludedTowers.begin(), excludedTowers.end(), pair<int, int>(ie, ip));
      if (exclude != excludedTowers.end()) {
        jettowers.push_back(index);
      }
    }  // initial input collection

  }  // pseudojets

  //
  // Create a new collections from the towers not included in jets
  //

  for (auto const& it : *fjInputs_) {
    int index = it.user_index();
    vector<int>::const_iterator itjet = find(jettowers.begin(), jettowers.end(), index);
    if (itjet == jettowers.end()) {
      const reco::CandidatePtr& originalTower = (*inputs_)[index];
      fastjet::PseudoJet orphan(originalTower->px(), originalTower->py(), originalTower->pz(), originalTower->energy());
      orphan.set_user_index(index);

      orphanInput.push_back(orphan);
    }
  }
}

double MultipleAlgoIterator::getEt(const reco::CandidatePtr& in) const {
  const CaloTower* ctc = dynamic_cast<const CaloTower*>(in.get());
  const GlobalPoint& pos = geo_->getPosition(ctc->id());
  double energy = ctc->emEnergy() + ctc->hadEnergy();
  double et = energy * sin(pos.theta());
  return et;
}

double MultipleAlgoIterator::getEta(const reco::CandidatePtr& in) const {
  const CaloTower* ctc = dynamic_cast<const CaloTower*>(in.get());
  const GlobalPoint& pos = geo_->getPosition(ctc->id());
  double eta = pos.eta();
  return eta;
}
