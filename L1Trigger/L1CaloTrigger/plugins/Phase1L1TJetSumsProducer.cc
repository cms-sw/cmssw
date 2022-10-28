// -*- C++ -*-
//
// Package: L1CaloTrigger
// Class: Phase1L1TJetSumsProducer
//
/**\class Phase1L1TJetSumsProducer Phase1L1TJetSumsProducer.cc L1Trigger/L1CaloTrigger/plugin/Phase1L1TJetSumsProducer.cc

Description: Computes HT and MHT from phase-1-like jets

*** INPUT PARAMETERS ***
  * sin/cosPhi: Value of sin/cos phi in the middle of each bin of the grid.
  * etaBinning: vdouble with eta binning (allows non-homogeneous binning in eta)
  * nBinsPhi: uint32, number of bins in phi
  * phiLow: double, min phi (typically -pi)
  * phiUp: double, max phi (typically +pi)
  * {m}htPtThreshold: Minimum jet pt for HT/MHT calculation
  * {m}htAbsEtaCut: 
  * pt/eta/philsb : lsb of quantities used in firmware implementation
  * outputCollectionName: string, tag for the output collection
   * inputCollectionTag: tag for input jet collection

*/
//
// Original Simone Bologna
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <cmath>

class Phase1L1TJetSumsProducer : public edm::one::EDProducer<edm::one::SharedResources> {
public:
  explicit Phase1L1TJetSumsProducer(const edm::ParameterSet&);
  ~Phase1L1TJetSumsProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // computes ht, adds jet pt to ht only if the pt of the jet is above the ht calculation threshold
  l1t::EtSum computeHT(const edm::Handle<std::vector<reco::CaloJet> > inputJets) const;

  // computes MHT
  // adds jet pt to mht only if the pt of the jet is above the mht calculation threshold
  // performs some calculations with digitised/integer quantities to ensure agreement with firmware
  l1t::EtSum computeMHT(const edm::Handle<std::vector<reco::CaloJet> > inputJets) const;

  const edm::EDGetTokenT<std::vector<reco::CaloJet> > inputJetCollectionTag_;

  // holds the sin and cos for HLs LUT emulation
  const std::vector<double> sinPhi_;
  const std::vector<double> cosPhi_;
  const unsigned int nBinsPhi_;

  // lower phi value
  const double phiLow_;
  // higher phi value
  const double phiUp_;
  // size of a phi bin
  const double phiStep_;
  // threshold for ht calculation
  const double htPtThreshold_;
  // threshold for ht calculation
  const double mhtPtThreshold_;
  // jet eta cut for ht calculation
  const double htAbsEtaCut_;
  // jet eta cut for mht calculation
  const double mhtAbsEtaCut_;
  // LSB of pt quantity
  const double ptlsb_;
  // label of sums
  const std::string outputCollectionName_;
};

// initialises plugin configuration and prepares ROOT file for saving the sums
Phase1L1TJetSumsProducer::Phase1L1TJetSumsProducer(const edm::ParameterSet& iConfig)
    : inputJetCollectionTag_{consumes<std::vector<reco::CaloJet> >(
          iConfig.getParameter<edm::InputTag>("inputJetCollectionTag"))},
      sinPhi_(iConfig.getParameter<std::vector<double> >("sinPhi")),
      cosPhi_(iConfig.getParameter<std::vector<double> >("cosPhi")),
      nBinsPhi_(iConfig.getParameter<unsigned int>("nBinsPhi")),
      phiLow_(iConfig.getParameter<double>("phiLow")),
      phiUp_(iConfig.getParameter<double>("phiUp")),
      phiStep_((phiUp_ - phiLow_) / nBinsPhi_),
      htPtThreshold_(iConfig.getParameter<double>("htPtThreshold")),
      mhtPtThreshold_(iConfig.getParameter<double>("mhtPtThreshold")),
      htAbsEtaCut_(iConfig.getParameter<double>("htAbsEtaCut")),
      mhtAbsEtaCut_(iConfig.getParameter<double>("mhtAbsEtaCut")),
      ptlsb_(iConfig.getParameter<double>("ptlsb")),
      outputCollectionName_(iConfig.getParameter<std::string>("outputCollectionName")) {
  usesResource("TFileService");
  produces<std::vector<l1t::EtSum> >(outputCollectionName_).setBranchAlias(outputCollectionName_);
}

void Phase1L1TJetSumsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const edm::Handle<std::vector<reco::CaloJet> >& jetCollectionHandle = iEvent.getHandle(inputJetCollectionTag_);

  // computing sums and storing them in sum object
  l1t::EtSum lHT = computeHT(jetCollectionHandle);
  l1t::EtSum lMHT = computeMHT(jetCollectionHandle);

  //packing sums in vector for event saving
  std::unique_ptr<std::vector<l1t::EtSum> > lSumVectorPtr(new std::vector<l1t::EtSum>(0));
  lSumVectorPtr->push_back(lHT);
  lSumVectorPtr->push_back(lMHT);
  iEvent.put(std::move(lSumVectorPtr), outputCollectionName_);

  return;
}

l1t::EtSum Phase1L1TJetSumsProducer::computeHT(const edm::Handle<std::vector<reco::CaloJet> > inputJets) const {
  double lHT = 0;
  for (const auto& jet : *inputJets) {
    double lJetPt = jet.pt();
    double lJetPhi = jet.phi();
    double lJetEta = jet.eta();
    if ((lJetPhi < phiLow_) || (lJetPhi >= phiUp_))
      continue;

    lHT += (lJetPt >= htPtThreshold_ && std::fabs(lJetEta) < htAbsEtaCut_) ? lJetPt : 0;
  }

  reco::Candidate::PolarLorentzVector lHTVector;
  lHTVector.SetPt(lHT);
  lHTVector.SetEta(0);
  lHTVector.SetPhi(0);
  l1t::EtSum lHTSum(lHTVector, l1t::EtSum::EtSumType::kTotalHt, 0, 0, 0, 0);
  return lHTSum;
}

l1t::EtSum Phase1L1TJetSumsProducer::computeMHT(const edm::Handle<std::vector<reco::CaloJet> > inputJets) const {
  int lTotalJetPx = 0;
  int lTotalJetPy = 0;

  std::vector<unsigned int> jetPtInPhiBins(nBinsPhi_, 0);

  for (const auto& jet : *inputJets) {
    double lJetPhi = jet.phi();

    if ((lJetPhi < phiLow_) || (lJetPhi >= phiUp_))
      continue;

    unsigned int iPhi = (lJetPhi - phiLow_) / phiStep_;

    if (jet.pt() >= mhtPtThreshold_ && std::fabs(jet.eta()) < mhtAbsEtaCut_) {
      unsigned int digiJetPt = floor(jet.pt() / ptlsb_);
      jetPtInPhiBins[iPhi] += digiJetPt;
    }
  }

  for (unsigned int iPhi = 0; iPhi < jetPtInPhiBins.size(); ++iPhi) {
    unsigned int digiJetPtSum = jetPtInPhiBins[iPhi];

    // retrieving sin cos from LUT emulator
    double lSinPhi = sinPhi_[iPhi];
    double lCosPhi = cosPhi_[iPhi];

    // checking if above threshold
    lTotalJetPx += trunc(digiJetPtSum * lCosPhi);
    lTotalJetPy += trunc(digiJetPtSum * lSinPhi);
  }

  double lMHT = floor(sqrt(lTotalJetPx * lTotalJetPx + lTotalJetPy * lTotalJetPy)) * ptlsb_;
  math::PtEtaPhiMLorentzVector lMHTVector(lMHT, 0, acos(lTotalJetPx / (lMHT / ptlsb_)), 0);
  l1t::EtSum lMHTSum(lMHTVector, l1t::EtSum::EtSumType::kMissingHt, 0, 0, 0, 0);

  return lMHTSum;
}

void Phase1L1TJetSumsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputJetCollectionTag",
                          edm::InputTag("l1tPhase1JetCalibrator", "Phase1L1TJetFromPfCandidates"));
  desc.add<std::vector<double> >("sinPhi");
  desc.add<std::vector<double> >("cosPhi");
  desc.add<unsigned int>("nBinsPhi", 72);
  desc.add<double>("phiLow", -M_PI);
  desc.add<double>("phiUp", M_PI);
  desc.add<double>("htPtThreshold", 30);
  desc.add<double>("mhtPtThreshold", 30);
  desc.add<double>("htAbsEtaCut", 3);
  desc.add<double>("mhtAbsEtaCut", 3);
  desc.add<double>("ptlsb", 0.25), desc.add<string>("outputCollectionName", "Sums");
  descriptions.add("Phase1L1TJetSumsProducer", desc);
}

// creates the plugin for later use in python
DEFINE_FWK_MODULE(Phase1L1TJetSumsProducer);
