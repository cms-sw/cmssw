// -*- C++ -*-
//
// Package: L1CaloTrigger
// Class: Phase1L1TJetProducer
//
/**\class Phase1L1TJetProducer Phase1L1TJetProducer.cc L1Trigger/L1CaloTrigger/plugin/Phase1L1TJetProducer.cc

Description: Produces jets with a phase-1 like sliding window algorithm using a collection of reco::Candidates in input.  Also calculates MET from the histogram used to find the jets.

*** INPUT PARAMETERS ***
  * etaBinning: vdouble with eta binning (allows non-homogeneous binning in eta)
  * nBinsPhi: uint32, number of bins in phi
  * phiLow: double, min phi (typically -pi)
  * phiUp: double, max phi (typically +pi)
  * jetIEtaSize: uint32, jet cluster size in ieta
  * jetIPhiSize: uint32, jet cluster size in iphi
  * trimmedGrid: Flag (bool) to remove three bins in each corner of grid in jet finding
  * seedPtThreshold: double, threshold of the seed tower
  * pt/eta/philsb : lsb of quantities used in firmware implementation
  * puSubtraction: bool, runs chunky doughnut pile-up subtraction, 9x9 jet only
  * eta/phiRegionEdges: Boundaries of the input (PF) regions
  * maxInputsPerRegion: Truncate number of inputes per input (PF) region
  * sin/cosPhi: Value of sin/cos phi in the middle of each bin of the grid.
  * met{HF}AbsETaCut: Eta selection of input candidates for calculation of MET
  * outputCollectionName: string, tag for the output collection
  * vetoZeroPt: bool, controls whether jets with 0 pt should be save. 
    It matters if PU is ON, as you can get negative or zero pt jets after it.
  * inputCollectionTag: inputtag, collection of reco::candidates used as input to the algo

*/
//
// Original Simone Bologna
// Created: Mon Jul 02 2018
// Modified 2020 Emyr Clement
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "TH2F.h"

#include <cmath>

#include <algorithm>
constexpr int x_scroll_min = -4;
constexpr int x_scroll_max = 4;
constexpr int y_scroll_min = 0;
constexpr int y_scroll_max = 3;

class Phase1L1TJetProducer : public edm::one::EDProducer<> {
public:
  explicit Phase1L1TJetProducer(const edm::ParameterSet&);
  ~Phase1L1TJetProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  /// Finds the seeds in the caloGrid, seeds are saved in a vector that contain the index in the TH2F of each seed
  std::vector<std::tuple<int, int>> findSeeds(float seedThreshold) const;

  // Calculates the pt sum of the bins around the seeds
  // And applies some PU mitigation
  // Warning - not used for some time, so user beware
  std::vector<reco::CaloJet> buildJetsFromSeedsWithPUSubtraction(const std::vector<std::tuple<int, int>>& seeds,
                                                                 bool killZeroPt) const;

  // Calculates the pt sum of the bins around the seeds
  std::vector<reco::CaloJet> buildJetsFromSeeds(const std::vector<std::tuple<int, int>>& seeds) const;

  // Used by buildJetsFromSeedsWithPUSubtraction
  // Implementation of pileup mitigation
  // Warning - not used for some time, so user beware
  void subtract9x9Pileup(reco::CaloJet& jet) const;

  /// Get the energy of a certain tower while correctly handling phi periodicity in case of overflow
  float getTowerEnergy(int iEta, int iPhi) const;

  // Implementation of pt sum of bins around one seed
  reco::CaloJet buildJetFromSeed(const std::tuple<int, int>& seed) const;

  // <3 handy method to fill the calogrid with whatever type
  template <class Container>
  void fillCaloGrid(TH2F& caloGrid, const Container& triggerPrimitives, const unsigned int regionIndex);

  // Digitise the eta and phi coordinates of input candidates
  // This converts the quantities to integers to reduce precision
  // And takes account of bin edge effects i.e. makes sure the
  // candidate ends up in the correct (i.e. same behaviour as the firmware) bin of caloGrid_
  std::pair<float, float> getCandidateDigiEtaPhi(const float eta,
                                                 const float phi,
                                                 const unsigned int regionIndex) const;

  // Sorts the input candidates into the PF regions they arrive in
  // Truncates the inputs.  Takes the first N candidates as they are provided, without any sorting (this may be needed in the future and/or provided in this way from emulation of layer 1)
  template <class Handle>
  std::vector<std::vector<reco::CandidatePtr>> prepareInputsIntoRegions(const Handle triggerPrimitives);

  // Converts phi and eta (PF) region indices to a single index
  unsigned int getRegionIndex(const unsigned int phiRegion, const unsigned int etaRegion) const;
  // From the single index, calculated by getRegionIndex, provides the lower eta and phi boundaries of the input (PF) region index
  std::pair<double, double> regionEtaPhiLowEdges(const unsigned int regionIndex) const;
  // From the single index, calculated by getRegionIndex, provides the upper eta and phi boundaries of the input (PF) region index
  std::pair<double, double> regionEtaPhiUpEdges(const unsigned int regionIndex) const;

  // computes MET
  // Takes grid used by jet finder and projects to 1D histogram of phi, bin contents are total pt in that phi bin
  // the phi bin index is used to retrieve the sin-cos value from the LUT emulator
  // the pt of the input is multiplied by that sin cos value to obtain px and py that is added to the total event px & py
  // after all the inputs have been processed we compute the total pt of the event, and set that as MET
  l1t::EtSum computeMET(const double etaCut, l1t::EtSum::EtSumType sumType) const;

  // Determine if this tower should be trimmed or not
  // Used only when trimmedGrid_ option is set to true
  // Trim means removing 3 towers in each corner of the square grid
  // giving a cross shaped grid, which is a bit more circular in shape than a square
  bool trimTower(const int etaIndex, const int phiIndex) const;

  edm::EDGetTokenT<edm::View<reco::Candidate>> inputCollectionTag_;
  // histogram containing our clustered inputs
  std::unique_ptr<TH2F> caloGrid_;

  std::vector<double> etaBinning_;
  size_t nBinsEta_;
  unsigned int nBinsPhi_;
  double phiLow_;
  double phiUp_;
  unsigned int jetIEtaSize_;
  unsigned int jetIPhiSize_;
  bool trimmedGrid_;
  double seedPtThreshold_;
  double ptlsb_;
  double philsb_;
  double etalsb_;
  bool puSubtraction_;
  bool vetoZeroPt_;
  // Eta and phi edges of input PF regions
  std::vector<double> etaRegionEdges_;
  std::vector<double> phiRegionEdges_;
  // Maximum number of candidates per input PF region
  unsigned int maxInputsPerRegion_;
  // LUT for sin and cos phi, to match that used in firmware
  std::vector<double> sinPhi_;
  std::vector<double> cosPhi_;
  // input eta cut for met calculation
  double metAbsEtaCut_;
  // input eta cut for metHF calculation
  double metHFAbsEtaCut_;
  std::string outputCollectionName_;
};

Phase1L1TJetProducer::Phase1L1TJetProducer(const edm::ParameterSet& iConfig)
    :  // getting configuration settings
      inputCollectionTag_{
          consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("inputCollectionTag"))},
      etaBinning_(iConfig.getParameter<std::vector<double>>("etaBinning")),
      nBinsEta_(etaBinning_.size() - 1),
      nBinsPhi_(iConfig.getParameter<unsigned int>("nBinsPhi")),
      phiLow_(iConfig.getParameter<double>("phiLow")),
      phiUp_(iConfig.getParameter<double>("phiUp")),
      jetIEtaSize_(iConfig.getParameter<unsigned int>("jetIEtaSize")),
      jetIPhiSize_(iConfig.getParameter<unsigned int>("jetIPhiSize")),
      trimmedGrid_(iConfig.getParameter<bool>("trimmedGrid")),
      seedPtThreshold_(iConfig.getParameter<double>("seedPtThreshold")),
      ptlsb_(iConfig.getParameter<double>("ptlsb")),
      philsb_(iConfig.getParameter<double>("philsb")),
      etalsb_(iConfig.getParameter<double>("etalsb")),
      puSubtraction_(iConfig.getParameter<bool>("puSubtraction")),
      vetoZeroPt_(iConfig.getParameter<bool>("vetoZeroPt")),
      etaRegionEdges_(iConfig.getParameter<std::vector<double>>("etaRegions")),
      phiRegionEdges_(iConfig.getParameter<std::vector<double>>("phiRegions")),
      maxInputsPerRegion_(iConfig.getParameter<unsigned int>("maxInputsPerRegion")),
      sinPhi_(iConfig.getParameter<std::vector<double>>("sinPhi")),
      cosPhi_(iConfig.getParameter<std::vector<double>>("cosPhi")),
      metAbsEtaCut_(iConfig.getParameter<double>("metAbsEtaCut")),
      metHFAbsEtaCut_(iConfig.getParameter<double>("metHFAbsEtaCut")),
      outputCollectionName_(iConfig.getParameter<std::string>("outputCollectionName")) {
  caloGrid_ =
      std::make_unique<TH2F>("caloGrid", "Calorimeter grid", nBinsEta_, etaBinning_.data(), nBinsPhi_, phiLow_, phiUp_);
  caloGrid_->GetXaxis()->SetTitle("#eta");
  caloGrid_->GetYaxis()->SetTitle("#phi");
  produces<std::vector<reco::CaloJet>>(outputCollectionName_).setBranchAlias(outputCollectionName_);
  produces<std::vector<l1t::EtSum>>(outputCollectionName_ + "MET").setBranchAlias(outputCollectionName_ + "MET");
}

Phase1L1TJetProducer::~Phase1L1TJetProducer() {}

float Phase1L1TJetProducer::getTowerEnergy(int iEta, int iPhi) const {
  // We return the pt of a certain bin in the calo grid, taking account of the phi periodicity when overflowing (e.g. phi > phiSize), and returning 0 for the eta out of bounds

  int nBinsEta = caloGrid_->GetNbinsX();
  int nBinsPhi = caloGrid_->GetNbinsY();
  while (iPhi < 1) {
    iPhi += nBinsPhi;
  }
  while (iPhi > nBinsPhi) {
    iPhi -= nBinsPhi;
  }
  if (iEta < 1) {
    return 0;
  }
  if (iEta > nBinsEta) {
    return 0;
  }
  return caloGrid_->GetBinContent(iEta, iPhi);
}

void Phase1L1TJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<reco::Candidate>> inputCollectionHandle;
  iEvent.getByToken(inputCollectionTag_, inputCollectionHandle);

  // sort inputs into PF regions
  std::vector<std::vector<reco::CandidatePtr>> inputsInRegions = prepareInputsIntoRegions<>(inputCollectionHandle);

  // histogramming the data
  caloGrid_->Reset();
  for (unsigned int iInputRegion = 0; iInputRegion < inputsInRegions.size(); ++iInputRegion) {
    fillCaloGrid<>(*(caloGrid_), inputsInRegions[iInputRegion], iInputRegion);
  }

  // find the seeds
  const auto& seedsVector = findSeeds(seedPtThreshold_);  // seedPtThreshold = 5
  // build jets from the seeds
  auto l1jetVector =
      puSubtraction_ ? buildJetsFromSeedsWithPUSubtraction(seedsVector, vetoZeroPt_) : buildJetsFromSeeds(seedsVector);

  // sort by pt
  std::sort(l1jetVector.begin(), l1jetVector.end(), [](const reco::CaloJet& jet1, const reco::CaloJet& jet2) {
    return jet1.pt() > jet2.pt();
  });

  auto l1jetVectorPtr = std::make_unique<std::vector<reco::CaloJet>>(l1jetVector);
  iEvent.put(std::move(l1jetVectorPtr), outputCollectionName_);

  // calculate METs
  l1t::EtSum lMET = computeMET(metAbsEtaCut_, l1t::EtSum::EtSumType::kMissingEt);
  l1t::EtSum lMETHF = computeMET(metHFAbsEtaCut_, l1t::EtSum::EtSumType::kMissingEtHF);
  std::unique_ptr<std::vector<l1t::EtSum>> lSumVectorPtr(new std::vector<l1t::EtSum>(0));
  lSumVectorPtr->push_back(lMET);
  lSumVectorPtr->push_back(lMETHF);
  iEvent.put(std::move(lSumVectorPtr), this->outputCollectionName_ + "MET");

  return;
}

void Phase1L1TJetProducer::subtract9x9Pileup(reco::CaloJet& jet) const {
  // these variables host the total pt in each sideband and the total pileup contribution
  float topBandPt = 0;
  float leftBandPt = 0;
  float rightBandPt = 0;
  float bottomBandPt = 0;
  float pileUpEnergy;

  // hold the jet's x-y (and z, as I have to use it, even if 2D) location in the histo
  int xCenter, yCenter, zCenter;
  // Retrieving histo-coords for seed
  caloGrid_->GetBinXYZ(caloGrid_->FindFixBin(jet.eta(), jet.phi()), xCenter, yCenter, zCenter);

  // Computing pileup
  for (int x = x_scroll_min; x <= x_scroll_max; x++) {
    for (int y = y_scroll_min; y < y_scroll_max; y++) {
      // top band, I go up 5 squares to reach the bottom of the top band
      // +x scrolls from left to right, +y scrolls up
      topBandPt += getTowerEnergy(xCenter + x, yCenter + (5 + y));
      // left band, I go left 5 squares (-5) to reach the bottom of the top band
      // +x scrolls from bottom to top, +y scrolls left
      leftBandPt += getTowerEnergy(xCenter - (5 + y), yCenter + x);
      // right band, I go right 5 squares (+5) to reach the bottom of the top band
      // +x scrolls from bottom to top, +y scrolls right
      rightBandPt += getTowerEnergy(xCenter + (5 + y), yCenter + x);
      // right band, I go right 5 squares (+5) to reach the bottom of the top band
      // +x scrolls from bottom to top, +y scrolls right
      bottomBandPt += getTowerEnergy(xCenter + x, yCenter - (5 + y));
    }
  }
  // adding bands and removing the maximum band (equivalent to adding the three minimum bands)
  pileUpEnergy = topBandPt + leftBandPt + rightBandPt + bottomBandPt -
                 std::max(topBandPt, std::max(leftBandPt, std::max(rightBandPt, bottomBandPt)));

  //preparing the new 4-momentum vector
  reco::Candidate::PolarLorentzVector ptVector;
  // removing pu contribution
  float ptAfterPUSubtraction = jet.pt() - pileUpEnergy;
  ptVector.SetPt((ptAfterPUSubtraction > 0) ? ptAfterPUSubtraction : 0);
  ptVector.SetEta(jet.eta());
  ptVector.SetPhi(jet.phi());
  //updating the jet
  jet.setP4(ptVector);
  jet.setPileup(pileUpEnergy);
  return;
}

std::vector<std::tuple<int, int>> Phase1L1TJetProducer::findSeeds(float seedThreshold) const {
  int nBinsX = caloGrid_->GetNbinsX();
  int nBinsY = caloGrid_->GetNbinsY();

  std::vector<std::tuple<int, int>> seeds;

  int etaHalfSize = (int)jetIEtaSize_ / 2;
  int phiHalfSize = (int)jetIPhiSize_ / 2;

  // for each point of the grid check if it is a local maximum
  // to do so I take a point, and look if is greater than the points around it (in the 9x9 neighborhood)
  // to prevent mutual exclusion, I check greater or equal for points above and right to the one I am considering (including the top-left point)
  // to prevent mutual exclusion, I check greater for points below and left to the one I am considering (including the bottom-right point)

  for (int iPhi = 1; iPhi <= nBinsY; iPhi++) {
    for (int iEta = 1; iEta <= nBinsX; iEta++) {
      float centralPt = caloGrid_->GetBinContent(iEta, iPhi);
      if (centralPt < seedThreshold)
        continue;
      bool isLocalMaximum = true;

      // Scanning through the grid centered on the seed
      for (int etaIndex = -etaHalfSize; etaIndex <= etaHalfSize; etaIndex++) {
        for (int phiIndex = -phiHalfSize; phiIndex <= phiHalfSize; phiIndex++) {
          if (trimmedGrid_) {
            if (trimTower(etaIndex, phiIndex))
              continue;
          }

          if ((etaIndex == 0) && (phiIndex == 0))
            continue;
          if (phiIndex > 0) {
            if (phiIndex > -etaIndex) {
              isLocalMaximum = ((isLocalMaximum) && (centralPt > getTowerEnergy(iEta + etaIndex, iPhi + phiIndex)));
            } else {
              isLocalMaximum = ((isLocalMaximum) && (centralPt >= getTowerEnergy(iEta + etaIndex, iPhi + phiIndex)));
            }
          } else {
            if (phiIndex >= -etaIndex) {
              isLocalMaximum = ((isLocalMaximum) && (centralPt > getTowerEnergy(iEta + etaIndex, iPhi + phiIndex)));
            } else {
              isLocalMaximum = ((isLocalMaximum) && (centralPt >= getTowerEnergy(iEta + etaIndex, iPhi + phiIndex)));
            }
          }
        }
      }
      if (isLocalMaximum) {
        seeds.emplace_back(iEta, iPhi);
      }
    }
  }

  return seeds;
}

reco::CaloJet Phase1L1TJetProducer::buildJetFromSeed(const std::tuple<int, int>& seed) const {
  int iEta = std::get<0>(seed);
  int iPhi = std::get<1>(seed);

  int etaHalfSize = (int)jetIEtaSize_ / 2;
  int phiHalfSize = (int)jetIPhiSize_ / 2;

  float ptSum = 0;
  // Scanning through the grid centered on the seed
  for (int etaIndex = -etaHalfSize; etaIndex <= etaHalfSize; etaIndex++) {
    for (int phiIndex = -phiHalfSize; phiIndex <= phiHalfSize; phiIndex++) {
      if (trimmedGrid_) {
        if (trimTower(etaIndex, phiIndex))
          continue;
      }
      ptSum += getTowerEnergy(iEta + etaIndex, iPhi + phiIndex);
    }
  }

  // Creating a jet with eta phi centered on the seed and momentum equal to the sum of the pt of the components
  reco::Candidate::PolarLorentzVector ptVector;
  ptVector.SetPt(ptSum);
  ptVector.SetEta(caloGrid_->GetXaxis()->GetBinCenter(iEta));
  ptVector.SetPhi(caloGrid_->GetYaxis()->GetBinCenter(iPhi));
  reco::CaloJet jet;
  jet.setP4(ptVector);
  return jet;
}

std::vector<reco::CaloJet> Phase1L1TJetProducer::buildJetsFromSeedsWithPUSubtraction(
    const std::vector<std::tuple<int, int>>& seeds, bool killZeroPt) const {
  // For each seed take a grid centered on the seed of the size specified by the user
  // Sum the pf in the grid, that will be the pt of the l1t jet. Eta and phi of the jet is taken from the seed.
  std::vector<reco::CaloJet> jets;
  for (const auto& seed : seeds) {
    reco::CaloJet jet = buildJetFromSeed(seed);
    subtract9x9Pileup(jet);
    //killing jets with 0 pt
    if ((vetoZeroPt_) && (jet.pt() <= 0))
      continue;
    jets.push_back(jet);
  }
  return jets;
}

std::vector<reco::CaloJet> Phase1L1TJetProducer::buildJetsFromSeeds(
    const std::vector<std::tuple<int, int>>& seeds) const {
  // For each seed take a grid centered on the seed of the size specified by the user
  // Sum the pf in the grid, that will be the pt of the l1t jet. Eta and phi of the jet is taken from the seed.
  std::vector<reco::CaloJet> jets;
  for (const auto& seed : seeds) {
    reco::CaloJet jet = buildJetFromSeed(seed);
    jets.push_back(jet);
  }
  return jets;
}

template <class Container>
void Phase1L1TJetProducer::fillCaloGrid(TH2F& caloGrid,
                                        const Container& triggerPrimitives,
                                        const unsigned int regionIndex) {
  //Filling the calo grid with the primitives
  for (const auto& primitiveIterator : triggerPrimitives) {
    // Get digitised (floating point with reduced precision) eta and phi
    std::pair<float, float> digi_EtaPhi =
        getCandidateDigiEtaPhi(primitiveIterator->eta(), primitiveIterator->phi(), regionIndex);

    caloGrid.Fill(static_cast<float>(digi_EtaPhi.first),
                  static_cast<float>(digi_EtaPhi.second),
                  static_cast<float>(primitiveIterator->pt()));
  }
}

std::pair<float, float> Phase1L1TJetProducer::getCandidateDigiEtaPhi(const float eta,
                                                                     const float phi,
                                                                     const unsigned int regionIndex) const {
  std::pair<double, double> regionLowEdges = regionEtaPhiLowEdges(regionIndex);

  int digitisedEta = floor((eta - regionLowEdges.second) / etalsb_);
  int digitisedPhi = floor((phi - regionLowEdges.first) / philsb_);

  // If eta or phi is on a bin edge
  // Put in bin above, to match behaviour of HLS
  // Unless it's on the last bin of this pf region
  // Then it is placed in the last bin, not the overflow
  TAxis* etaAxis = caloGrid_->GetXaxis();
  std::pair<double, double> regionUpEdges = regionEtaPhiUpEdges(regionIndex);
  int digiEtaEdgeLastBinUp = floor((regionUpEdges.second - regionLowEdges.second) / etalsb_);
  // If the digi eta is outside the last bin of this pf region
  // Set the digitised quantity so it would be in the last bin
  // These cases could be avoided by sorting input candidates based on digitised eta/phi
  if (digitisedEta >= digiEtaEdgeLastBinUp) {
    digitisedEta = digiEtaEdgeLastBinUp - 1;
  } else {
    for (int i = 0; i < etaAxis->GetNbins(); ++i) {
      if (etaAxis->GetBinUpEdge(i) < regionLowEdges.second)
        continue;
      int digiEdgeBinUp = floor((etaAxis->GetBinUpEdge(i) - regionLowEdges.second) / etalsb_);
      if (digiEdgeBinUp == digitisedEta) {
        digitisedEta += 1;
      }
    }
  }

  // Similar for phi
  TAxis* phiAxis = caloGrid_->GetYaxis();
  int digiPhiEdgeLastBinUp = floor((regionUpEdges.first - regionLowEdges.first) / philsb_);
  if (digitisedPhi >= digiPhiEdgeLastBinUp) {
    digitisedPhi = digiPhiEdgeLastBinUp - 1;
  } else {
    for (int i = 0; i < phiAxis->GetNbins(); ++i) {
      if (phiAxis->GetBinUpEdge(i) < regionLowEdges.first)
        continue;
      int digiEdgeBinUp = floor((phiAxis->GetBinUpEdge(i) - regionLowEdges.first) / philsb_);
      if (digiEdgeBinUp == digitisedPhi) {
        digitisedPhi += 1;
      }
    }
  }

  // Convert digitised eta and phi back to floating point quantities with reduced precision
  float floatDigitisedEta = (digitisedEta + 0.5) * etalsb_ + regionLowEdges.second;
  float floatDigitisedPhi = (digitisedPhi + 0.5) * philsb_ + regionLowEdges.first;

  return std::pair<float, float>{floatDigitisedEta, floatDigitisedPhi};
}

void Phase1L1TJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputCollectionTag", edm::InputTag("l1pfCandidates", "Puppi"));
  desc.add<std::vector<double>>("etaBinning");
  desc.add<unsigned int>("nBinsPhi", 72);
  desc.add<double>("phiLow", -M_PI);
  desc.add<double>("phiUp", M_PI);
  desc.add<unsigned int>("jetIEtaSize", 7);
  desc.add<unsigned int>("jetIPhiSize", 7);
  desc.add<bool>("trimmedGrid", false);
  desc.add<double>("seedPtThreshold", 5);
  desc.add<double>("ptlsb", 0.25), desc.add<double>("philsb", 0.0043633231), desc.add<double>("etalsb", 0.0043633231),
      desc.add<bool>("puSubtraction", false);
  desc.add<string>("outputCollectionName", "UncalibratedPhase1L1TJetFromPfCandidates");
  desc.add<bool>("vetoZeroPt", true);
  desc.add<std::vector<double>>("etaRegions");
  desc.add<std::vector<double>>("phiRegions");
  desc.add<unsigned int>("maxInputsPerRegion", 18);
  desc.add<std::vector<double>>("sinPhi");
  desc.add<std::vector<double>>("cosPhi");
  desc.add<double>("metAbsEtaCut", 3);
  desc.add<double>("metHFAbsEtaCut", 5);
  descriptions.add("Phase1L1TJetProducer", desc);
}

template <class Handle>
std::vector<std::vector<edm::Ptr<reco::Candidate>>> Phase1L1TJetProducer::prepareInputsIntoRegions(
    const Handle triggerPrimitives) {
  std::vector<std::vector<reco::CandidatePtr>> inputsInRegions{etaRegionEdges_.size() * (phiRegionEdges_.size() - 1)};

  for (unsigned int i = 0; i < triggerPrimitives->size(); ++i) {
    reco::CandidatePtr tp(triggerPrimitives, i);

    if (tp->phi() < phiRegionEdges_.front() || tp->phi() >= phiRegionEdges_.back() ||
        tp->eta() < etaRegionEdges_.front() || tp->eta() >= etaRegionEdges_.back())
      continue;

    // Which phi region does this tp belong to
    auto it_phi = phiRegionEdges_.begin();
    it_phi = std::upper_bound(phiRegionEdges_.begin(), phiRegionEdges_.end(), tp->phi()) - 1;

    // Which eta region does this tp belong to
    auto it_eta = etaRegionEdges_.begin();
    it_eta = std::upper_bound(etaRegionEdges_.begin(), etaRegionEdges_.end(), tp->eta()) - 1;

    if (it_phi != phiRegionEdges_.end() && it_eta != etaRegionEdges_.end()) {
      auto phiRegion = it_phi - phiRegionEdges_.begin();
      auto etaRegion = it_eta - etaRegionEdges_.begin();
      inputsInRegions[getRegionIndex(phiRegion, etaRegion)].emplace_back(tp);
    }
  }

  // Truncate number of inputs in each pf region
  for (auto& inputs : inputsInRegions) {
    if (inputs.size() > maxInputsPerRegion_) {
      inputs.resize(maxInputsPerRegion_);
    }
  }

  return inputsInRegions;
}

unsigned int Phase1L1TJetProducer::getRegionIndex(const unsigned int phiRegion, const unsigned int etaRegion) const {
  return etaRegion * (phiRegionEdges_.size() - 1) + phiRegion;
}

std::pair<double, double> Phase1L1TJetProducer::regionEtaPhiLowEdges(const unsigned int regionIndex) const {
  unsigned int phiRegion = regionIndex % (phiRegionEdges_.size() - 1);
  unsigned int etaRegion = (regionIndex - phiRegion) / (phiRegionEdges_.size() - 1);
  return std::pair<double, double>{phiRegionEdges_.at(phiRegion), etaRegionEdges_.at(etaRegion)};
}

std::pair<double, double> Phase1L1TJetProducer::regionEtaPhiUpEdges(const unsigned int regionIndex) const {
  unsigned int phiRegion = regionIndex % (phiRegionEdges_.size() - 1);
  unsigned int etaRegion = (regionIndex - phiRegion) / (phiRegionEdges_.size() - 1);
  if (phiRegion == phiRegionEdges_.size() - 1) {
    return std::pair<double, double>{phiRegionEdges_.at(phiRegion), etaRegionEdges_.at(etaRegion + 1)};
  } else if (etaRegion == etaRegionEdges_.size() - 1) {
    return std::pair<double, double>{phiRegionEdges_.at(phiRegion + 1), etaRegionEdges_.at(etaRegion)};
  }

  return std::pair<double, double>{phiRegionEdges_.at(phiRegion + 1), etaRegionEdges_.at(etaRegion + 1)};
}

l1t::EtSum Phase1L1TJetProducer::computeMET(const double etaCut, l1t::EtSum::EtSumType sumType) const {
  const auto lowEtaBin = caloGrid_->GetXaxis()->FindBin(-1.0 * etaCut);
  const auto highEtaBin = caloGrid_->GetXaxis()->FindBin(etaCut) - 1;
  const auto phiProjection = caloGrid_->ProjectionY("temp", lowEtaBin, highEtaBin);

  // Use digitised quantities when summing to improve agreement with firmware
  int totalDigiPx{0};
  int totalDigiPy{0};

  for (int i = 1; i < phiProjection->GetNbinsX() + 1; ++i) {
    double pt = phiProjection->GetBinContent(i);
    totalDigiPx += trunc(floor(pt / ptlsb_) * cosPhi_[i - 1]);
    totalDigiPy += trunc(floor(pt / ptlsb_) * sinPhi_[i - 1]);
  }

  double lMET = floor(sqrt(totalDigiPx * totalDigiPx + totalDigiPy * totalDigiPy)) * ptlsb_;

  math::PtEtaPhiMLorentzVector lMETVector(lMET, 0, acos(totalDigiPx / (lMET / ptlsb_)), 0);
  l1t::EtSum lMETSum(lMETVector, sumType, 0, 0, 0, 0);
  return lMETSum;
}

bool Phase1L1TJetProducer::trimTower(const int etaIndex, const int phiIndex) const {
  int etaHalfSize = jetIEtaSize_ / 2;
  int phiHalfSize = jetIPhiSize_ / 2;

  if (etaIndex == -etaHalfSize || etaIndex == etaHalfSize) {
    if (phiIndex <= -phiHalfSize + 1 || phiIndex >= phiHalfSize - 1) {
      return true;
    }
  } else if (etaIndex == -etaHalfSize + 1 || etaIndex == etaHalfSize - 1) {
    if (phiIndex == -phiHalfSize || phiIndex == phiHalfSize) {
      return true;
    }
  }

  return false;
}
DEFINE_FWK_MODULE(Phase1L1TJetProducer);
