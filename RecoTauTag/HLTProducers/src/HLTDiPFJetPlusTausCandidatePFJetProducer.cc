/** Description: Check correlation between PFJet pairs and filtered PFTau pairs and store the PFJet pairs.
For (j1, j2, t1, t2) where j1, j2 from the PFJet collection and t1, t2 from the filtered PFTau collection,
the module checks if there is no overlap (within dRmin) between j1, j2, t1, t2, i.e. there are 4 different objects.
In addition, the module imposes the following cuts:
* mjjMin: the min invariant mass cut on (j1, j2)
* extraTauPtCut: the leading tau pt cut on (t1, t2) (under the assumption t1, t2 are products of a subleading pt filter with minN = 2)
The module stores j1, j2 of any (j1, j2, t1, t2) that satisfies the conditions above. */

/* Extended for the case of j1, j2, t1 (no t2, i.e. there are only 3 different objects)
 */

#include <algorithm>
#include <memory>
#include <set>
#include <utility>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "CommonTools/Utils/interface/PtComparator.h"

class HLTDiPFJetPlusTausCandidatePFJetProducer : public edm::global::EDProducer<> {
public:
  explicit HLTDiPFJetPlusTausCandidatePFJetProducer(const edm::ParameterSet&);
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tauSrc_;
  const edm::EDGetTokenT<reco::PFJetCollection> pfJetSrc_;
  const double extraTauPtCut_;
  const double mjjMin_, m2jjMin_;
  const double dRmin_, dRmin2_;
  GreaterByPt<reco::PFJet> pTComparator_;  // pt comparator
};

HLTDiPFJetPlusTausCandidatePFJetProducer::HLTDiPFJetPlusTausCandidatePFJetProducer(const edm::ParameterSet& iConfig)
    : tauSrc_(consumes(iConfig.getParameter<edm::InputTag>("tauSrc"))),
      pfJetSrc_(consumes(iConfig.getParameter<edm::InputTag>("pfJetSrc"))),
      extraTauPtCut_(iConfig.getParameter<double>("extraTauPtCut")),
      mjjMin_(iConfig.getParameter<double>("mjjMin")),
      m2jjMin_(mjjMin_ * mjjMin_),
      dRmin_(iConfig.getParameter<double>("dRmin")),
      dRmin2_(dRmin_ * dRmin_) {
  if (dRmin_ <= 0.) {
    throw cms::Exception("HLTDiPFJetPlusTausCandidatePFJetProducer")
        << "invalid value for parameter \"dRmin\" (must be > 0): " << dRmin_;
  }
  produces<reco::PFJetCollection>();
}

void HLTDiPFJetPlusTausCandidatePFJetProducer::produce(edm::StreamID,
                                                       edm::Event& iEvent,
                                                       const edm::EventSetup&) const {
  auto const& pfJets = iEvent.get(pfJetSrc_);

  auto cleanedPFJets = std::make_unique<reco::PFJetCollection>();
  cleanedPFJets->reserve(pfJets.size());

  trigger::VRpftau taus;
  iEvent.get(tauSrc_).getObjects(trigger::TriggerTau, taus);

  std::set<unsigned int> indices;

  for (unsigned int iJet1 = 0; iJet1 < pfJets.size(); iJet1++) {
    for (unsigned int iJet2 = iJet1 + 1; iJet2 < pfJets.size(); iJet2++) {
      bool correctComb = false;
      const reco::PFJet& myPFJet1 = pfJets[iJet1];
      const reco::PFJet& myPFJet2 = pfJets[iJet2];

      if (mjjMin_ >= 0. && (myPFJet1.p4() + myPFJet2.p4()).M2() < m2jjMin_)
        continue;

      for (unsigned int iTau1 = 0; iTau1 < taus.size(); iTau1++) {
        if (reco::deltaR2(taus[iTau1]->p4(), myPFJet1.p4()) < dRmin2_)
          continue;
        if (reco::deltaR2(taus[iTau1]->p4(), myPFJet2.p4()) < dRmin2_)
          continue;

        if (taus.size() == 1) {
          if (taus[iTau1]->pt() < extraTauPtCut_)
            continue;

          correctComb = true;
        } else {
          for (unsigned int iTau2 = iTau1 + 1; iTau2 < taus.size(); iTau2++) {
            if (taus[iTau1]->pt() < extraTauPtCut_ && taus[iTau2]->pt() < extraTauPtCut_)
              continue;

            if (reco::deltaR2(taus[iTau2]->p4(), myPFJet1.p4()) < dRmin2_)
              continue;
            if (reco::deltaR2(taus[iTau2]->p4(), myPFJet2.p4()) < dRmin2_)
              continue;

            correctComb = true;
            break;
          }
        }
        if (correctComb)
          break;
      }

      if (correctComb) {
        indices.insert(iJet1);
        indices.insert(iJet2);
      }
    }

    for (const auto& i : indices)
      cleanedPFJets->emplace_back(pfJets[i]);
  }
  // sort jets in pt
  std::sort(cleanedPFJets->begin(), cleanedPFJets->end(), pTComparator_);
  iEvent.put(std::move(cleanedPFJets));
}

void HLTDiPFJetPlusTausCandidatePFJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pfJetSrc", edm::InputTag("hltAK4PFJetsCorrected"))->setComment("Input collection of PFJets");
  desc.add<edm::InputTag>("tauSrc", edm::InputTag("hltSinglePFTau20TrackPt1LooseChargedIsolationReg"))
      ->setComment("Input collection of PFTaus that have passed ID and isolation requirements");
  desc.add<double>("extraTauPtCut", 45)->setComment("In case of asymmetric tau pt cuts");
  desc.add<double>("mjjMin", 500)->setComment("VBF dijet mass condition");
  desc.add<double>("dRmin", 0.5)->setComment("Minimum dR between PFJets and filtered PFTaus");
  descriptions.setComment(
      "This module produces a collection of PFJets that are cross-cleaned with respect to PFTaus passing a HLT "
      "filter.");
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTDiPFJetPlusTausCandidatePFJetProducer);
