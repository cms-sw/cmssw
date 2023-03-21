// -*- C++ -*-
//
// Package:    RecoTauTag/HLTProducers
// Class:      L1TJetsMatching
//
/**\class L1TJetsMatching L1TJetsMatching.h 
 RecoTauTag/HLTProducers/interface/L1TJetsMatching.h
 Description: 
 Matching L1 to PF/Calo Jets. Used for HLT_VBF paths.
	*Matches PF/Calo Jets to L1 jets from the dedicated seed
	*Adds selection criteria to the leading/subleading jets as well as the maximum dijet mass
	*Separates collections of PF/Calo jets into two categories
 
 
*/
//
// Original Author:  Vukasin Milosevic
//         Created:  Thu, 01 Jun 2017 17:23:00 GMT
//
//

#ifndef RecoTauTag_HLTProducers_L1TJetsMatching_h
#define RecoTauTag_HLTProducers_L1TJetsMatching_h

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <vector>
#include <utility>
#include <tuple>
#include <string>

template <typename T>
class L1TJetsMatching : public edm::global::EDProducer<> {
public:
  explicit L1TJetsMatching(const edm::ParameterSet&);
  ~L1TJetsMatching() override = default;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::pair<std::vector<T>, std::vector<T>> categorise(
      const std::vector<T>& pfMatchedJets, double pt1, double pt2, double pt3, double Mjj) const;
  std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> categoriseVBFPlus2CentralJets(
      const std::vector<T>& pfMatchedJets, double pt1, double pt2, double pt3, double Mjj) const;
  const edm::EDGetTokenT<std::vector<T>> jetSrc_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> jetTrigger_;
  const std::string matchingMode_;
  const double pt1Min_;
  const double pt2Min_;
  const double pt3Min_;
  const double mjjMin_;
  const double matchingR2_;
};
//
// class declaration
//
template <typename T>
std::pair<std::vector<T>, std::vector<T>> L1TJetsMatching<T>::categorise(
    const std::vector<T>& pfMatchedJets, double pt1, double pt2, double pt3, double Mjj) const {
  std::pair<std::vector<T>, std::vector<T>> output;
  unsigned int i1 = 0;
  unsigned int i2 = 0;
  double m2jj = 0;
  if (pfMatchedJets.size() > 1) {
    for (unsigned int i = 0; i < pfMatchedJets.size() - 1; i++) {
      const T& myJet1 = (pfMatchedJets)[i];

      for (unsigned int j = i + 1; j < pfMatchedJets.size(); j++) {
        const T& myJet2 = (pfMatchedJets)[j];

        const double m2jj_test = (myJet1.p4() + myJet2.p4()).M2();

        if (m2jj_test > m2jj) {
          m2jj = m2jj_test;
          i1 = i;
          i2 = j;
        }
      }
    }

    const T& myJet1 = (pfMatchedJets)[i1];
    const T& myJet2 = (pfMatchedJets)[i2];
    const double M2jj = (Mjj >= 0. ? Mjj * Mjj : -1.);

    if ((m2jj > M2jj) && (myJet1.pt() >= pt1) && (myJet2.pt() > pt2)) {
      output.first.push_back(myJet1);
      output.first.push_back(myJet2);
    }

    if ((m2jj > M2jj) && (myJet1.pt() < pt3) && (myJet1.pt() > pt2) && (myJet2.pt() > pt2)) {
      const T& myJetTest = (pfMatchedJets)[0];
      if (myJetTest.pt() > pt3) {
        output.second.push_back(myJet1);
        output.second.push_back(myJet2);
        output.second.push_back(myJetTest);
      }
    }
  }

  return output;
}
template <typename T>
std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> L1TJetsMatching<T>::categoriseVBFPlus2CentralJets(
    const std::vector<T>& pfMatchedJets, double pt1, double pt2, double pt3, double Mjj) const {  //60, 30, 50, 500
  std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> output;
  unsigned int i1 = 0;
  unsigned int i2 = 0;

  double m2jj = 0;
  if (pfMatchedJets.size() > 1) {
    for (unsigned int i = 0; i < pfMatchedJets.size() - 1; i++) {
      const T& myJet1 = (pfMatchedJets)[i];

      for (unsigned int j = i + 1; j < pfMatchedJets.size(); j++) {
        const T& myJet2 = (pfMatchedJets)[j];

        const double m2jj_test = (myJet1.p4() + myJet2.p4()).M2();

        if (m2jj_test > m2jj) {
          m2jj = m2jj_test;
          i1 = i;
          i2 = j;
        }
      }
    }

    const T& myJet1 = (pfMatchedJets)[i1];
    const T& myJet2 = (pfMatchedJets)[i2];
    const double M2jj = (Mjj >= 0. ? Mjj * Mjj : -1.);

    std::vector<T> vec4jets;
    vec4jets.reserve(4);
    std::vector<T> vec5jets;
    vec5jets.reserve(5);
    std::vector<T> vec6jets;
    vec6jets.reserve(6);
    if (pfMatchedJets.size() > 3) {
      if ((m2jj > M2jj) && (myJet1.pt() >= pt3) && (myJet2.pt() > pt2)) {
        vec4jets.push_back(myJet1);
        vec4jets.push_back(myJet2);

        for (unsigned int i = 0; i < pfMatchedJets.size(); i++) {
          if (vec4jets.size() > 3)
            break;
          if (i == i1 or i == i2)
            continue;
          vec4jets.push_back(pfMatchedJets[i]);
        }
      }

      if ((m2jj > M2jj) && (myJet1.pt() < pt1) && (myJet1.pt() < pt3) && (myJet1.pt() > pt2) &&
          (myJet2.pt() > pt2)) {  //60, 30, 50, 500

        std::vector<unsigned int> idx_jets;
        idx_jets.reserve(pfMatchedJets.size() - 2);

        for (unsigned int i = 0; i < pfMatchedJets.size(); i++) {
          if (i == i1 || i == i2)
            continue;
          if (pfMatchedJets[i].pt() > pt2) {
            idx_jets.push_back(i);
          }
        }
        if (idx_jets.size() == 3) {
          vec5jets.push_back(myJet1);
          vec5jets.push_back(myJet2);
          vec5jets.push_back(pfMatchedJets[idx_jets[0]]);
          vec5jets.push_back(pfMatchedJets[idx_jets[1]]);
          vec5jets.push_back(pfMatchedJets[idx_jets[2]]);

        } else if (idx_jets.size() > 3) {
          vec6jets.push_back(myJet1);
          vec6jets.push_back(myJet2);
          vec6jets.push_back(pfMatchedJets[idx_jets[0]]);
          vec6jets.push_back(pfMatchedJets[idx_jets[1]]);
          vec6jets.push_back(pfMatchedJets[idx_jets[2]]);
          vec6jets.push_back(pfMatchedJets[idx_jets[3]]);
        }
      }
    }

    output = std::make_tuple(vec4jets, vec5jets, vec6jets);
  }

  return output;
}

template <typename T>
L1TJetsMatching<T>::L1TJetsMatching(const edm::ParameterSet& iConfig)
    : jetSrc_(consumes<std::vector<T>>(iConfig.getParameter<edm::InputTag>("JetSrc"))),
      jetTrigger_(consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<edm::InputTag>("L1JetTrigger"))),
      matchingMode_(iConfig.getParameter<std::string>("matchingMode")),
      pt1Min_(iConfig.getParameter<double>("pt1Min")),
      pt2Min_(iConfig.getParameter<double>("pt2Min")),
      pt3Min_(iConfig.getParameter<double>("pt3Min")),
      mjjMin_(iConfig.getParameter<double>("mjjMin")),
      matchingR2_(iConfig.getParameter<double>("matchingR") * iConfig.getParameter<double>("matchingR")) {
  if (matchingMode_ == "VBF") {  // Default
    produces<std::vector<T>>("TwoJets");
    produces<std::vector<T>>("ThreeJets");
  } else if (matchingMode_ == "VBFPlus2CentralJets") {
    produces<std::vector<T>>("FourJets");
    produces<std::vector<T>>("FiveJets");
    produces<std::vector<T>>("SixJets");
  } else {
    throw cms::Exception("InvalidConfiguration") << "invalid value for \"matchingMode\": " << matchingMode_
                                                 << " (valid values are \"VBF\" and \"VBFPlus2CentralJets\")";
  }
}

template <typename T>
void L1TJetsMatching<T>::produce(edm::StreamID iSId, edm::Event& iEvent, const edm::EventSetup& iES) const {
  unique_ptr<std::vector<T>> pfMatchedJets(new std::vector<T>);

  // Getting HLT jets to be matched
  edm::Handle<std::vector<T>> pfJets;
  iEvent.getByToken(jetSrc_, pfJets);

  edm::Handle<trigger::TriggerFilterObjectWithRefs> l1TriggeredJets;
  iEvent.getByToken(jetTrigger_, l1TriggeredJets);

  //l1t::TauVectorRef jetCandRefVec;
  l1t::JetVectorRef jetCandRefVec;
  l1TriggeredJets->getObjects(trigger::TriggerL1Jet, jetCandRefVec);

  math::XYZPoint a(0., 0., 0.);

  //std::cout<<"PFsize= "<<pfJets->size()<<endl<<" L1size= "<<jetCandRefVec.size()<<std::endl;
  for (unsigned int iJet = 0; iJet < pfJets->size(); iJet++) {
    const T& myJet = (*pfJets)[iJet];
    for (unsigned int iL1Jet = 0; iL1Jet < jetCandRefVec.size(); iL1Jet++) {
      // Find the relative L2pfJets, to see if it has been reconstructed
      //  if ((iJet<3) && (iL1Jet==0))  std::cout<<myJet.p4().Pt()<<" ";
      if ((reco::deltaR2(myJet.p4(), jetCandRefVec[iL1Jet]->p4()) < matchingR2_) && (myJet.pt() > pt2Min_)) {
        pfMatchedJets->push_back(myJet);
        break;
      }
    }
  }
  // order pfMatchedJets by pT
  std::sort(pfMatchedJets->begin(), pfMatchedJets->end(), [](const T& j1, const T& j2) { return j1.pt() > j2.pt(); });

  if (matchingMode_ == "VBF") {  // Default
    std::pair<std::vector<T>, std::vector<T>> output = categorise(*pfMatchedJets, pt1Min_, pt2Min_, pt3Min_, mjjMin_);
    auto output1 = std::make_unique<std::vector<T>>(output.first);
    auto output2 = std::make_unique<std::vector<T>>(output.second);

    iEvent.put(std::move(output1), "TwoJets");
    iEvent.put(std::move(output2), "ThreeJets");

  } else if (matchingMode_ == "VBFPlus2CentralJets") {
    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> output =
        categoriseVBFPlus2CentralJets(*pfMatchedJets, pt1Min_, pt2Min_, pt3Min_, mjjMin_);
    auto output1 = std::make_unique<std::vector<T>>(std::get<0>(output));
    auto output2 = std::make_unique<std::vector<T>>(std::get<1>(output));
    auto output3 = std::make_unique<std::vector<T>>(std::get<2>(output));

    iEvent.put(std::move(output1), "FourJets");
    iEvent.put(std::move(output2), "FiveJets");
    iEvent.put(std::move(output3), "SixJets");
  }
}

template <typename T>
void L1TJetsMatching<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("L1JetTrigger", edm::InputTag("hltL1DiJetVBF"))->setComment("Name of trigger filter");
  desc.add<edm::InputTag>("JetSrc", edm::InputTag("hltAK4PFJetsTightIDCorrected"))
      ->setComment("Input collection of PFJets");
  desc.add<std::string>("matchingMode", "VBF")
      ->setComment("Switch from Di/tri-jet (VBF) to Multi-jet (VBFPlus2CentralJets) matching");
  desc.add<double>("pt1Min", 110.0)->setComment("Minimal pT1 of PFJets to match");
  desc.add<double>("pt2Min", 35.0)->setComment("Minimal pT2 of PFJets to match");
  desc.add<double>("pt3Min", 110.0)->setComment("Minimum pT3 of PFJets to match");
  desc.add<double>("mjjMin", 650.0)->setComment("Minimal mjj of matched PFjets");
  desc.add<double>("matchingR", 0.5)->setComment("dR value used for matching");
  descriptions.setComment(
      "This module produces collection of PFJets matched to L1 Taus / Jets passing a HLT filter (Only p4 and vertex "
      "of returned PFJets are set).");
  descriptions.add(defaultModuleLabel<L1TJetsMatching<T>>(), desc);
}

#endif
