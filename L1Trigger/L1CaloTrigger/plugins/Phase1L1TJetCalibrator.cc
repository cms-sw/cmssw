// -*- C++ -*-
//
// Package:    L1CaloTrigger
// Class:      Phase1L1TJetCalibrator
//
/**\class Phase1L1TJetCalibrator Phase1L1TJetCalibrator.cc 

 Description: Applies calibrations to reco::calojets

 *** INPUT PARAMETERS ***
   * inputCollectionTag, InputTag, collection of reco calojet to calibrate
   * absEtaBinning, vdouble with eta bins, allows for non-homogeneous binning
   * calibration, VPSet with calibration factors
   * outputCollectionName, string, output collection tag name

*/
//
// Original Author:  Simone Bologna
//         Created:  Wed, 19 Dec 2018 12:44:23 GMT
//
// Rewrite: Vladimir Rekovic, Oct 2020
//

// system include files
#include <memory>

#include <algorithm>
#include <cmath>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

class Phase1L1TJetCalibrator : public edm::stream::EDProducer<> {
public:
  explicit Phase1L1TJetCalibrator(const edm::ParameterSet&);
  ~Phase1L1TJetCalibrator() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<std::vector<reco::CaloJet>> inputCollectionTag_;
  std::vector<double> absEtaBinning_;
  size_t nBinsEta_;
  std::vector<edm::ParameterSet> calibration_;
  std::string outputCollectionName_;

  std::vector<std::vector<double>> jetCalibrationFactorsBinnedInEta_;
  std::vector<std::vector<double>> _jetCalibrationFactorsPtBins;

  // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
Phase1L1TJetCalibrator::Phase1L1TJetCalibrator(const edm::ParameterSet& iConfig)
    : inputCollectionTag_(edm::EDGetTokenT<std::vector<reco::CaloJet>>(
          consumes<std::vector<reco::CaloJet>>(iConfig.getParameter<edm::InputTag>("inputCollectionTag")))),
      absEtaBinning_(iConfig.getParameter<std::vector<double>>("absEtaBinning")),
      nBinsEta_(absEtaBinning_.size() - 1),
      calibration_(iConfig.getParameter<std::vector<edm::ParameterSet>>("calibration")),
      outputCollectionName_(iConfig.getParameter<std::string>("outputCollectionName"))

{
  for (const auto& pset : calibration_) {
    _jetCalibrationFactorsPtBins.emplace_back(pset.getParameter<std::vector<double>>("l1tPtBins"));
    jetCalibrationFactorsBinnedInEta_.emplace_back(pset.getParameter<std::vector<double>>("l1tCalibrationFactors"));
  }

  produces<std::vector<reco::CaloJet>>(outputCollectionName_).setBranchAlias(outputCollectionName_);
}

Phase1L1TJetCalibrator::~Phase1L1TJetCalibrator() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void Phase1L1TJetCalibrator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<std::vector<reco::CaloJet>> inputCollectionHandle;
  iEvent.getByToken(inputCollectionTag_, inputCollectionHandle);

  auto calibratedCollectionPtr = std::make_unique<std::vector<reco::CaloJet>>();

  // for each candidate:
  // 1 get pt and eta
  // 2 run a dicotomic search on the eta vector to find the eta index
  // 3 fetch the corresponding calibration elements
  // 4 run a dicotomic search on the pt binning to find the pt index
  // 5 fetch the calibration factor
  // 6 update the candidate pt by applying the calibration factor
  // 7 store calibrated candidate in a new collection

  calibratedCollectionPtr->reserve(inputCollectionHandle->size());

  for (const auto& candidate : *inputCollectionHandle) {
    // 1
    float pt = candidate.pt();
    float eta = candidate.eta();

    //2
    auto etaBin = upper_bound(absEtaBinning_.begin(), absEtaBinning_.end(), fabs(eta));
    int etaIndex = etaBin - absEtaBinning_.begin() - 1;

    //3
    const std::vector<double>& l1tPtBins = _jetCalibrationFactorsPtBins[etaIndex];
    const std::vector<double>& l1tCalibrationFactors = jetCalibrationFactorsBinnedInEta_[etaIndex];

    //4
    auto ptBin = upper_bound(l1tPtBins.begin(), l1tPtBins.end(), pt);
    int ptBinIndex = ptBin - l1tPtBins.begin() - 1;

    //5
    const double& l1tCalibrationFactor = l1tCalibrationFactors[ptBinIndex];

    //6 and 7
    reco::Candidate::PolarLorentzVector candidateP4(candidate.polarP4());
    candidateP4.SetPt(candidateP4.pt() * l1tCalibrationFactor);
    calibratedCollectionPtr->emplace_back(candidate);
    calibratedCollectionPtr->back().setP4(candidateP4);

#ifdef DEBUG
    if (newCandidate->pt() < 0) {
      LogDebug("Phase1L1TJetCalibrator") << "######################" << std::endl;
      LogDebug("Phase1L1TJetCalibrator") << "PRE-CALIBRATION " << std::endl;
      LogDebug("Phase1L1TJetCalibrator") << "\t Jet properties (pt, eta, phi, pile-up): " << candidate.pt() << "\t"
                                         << candidate.eta() << "\t" LogDebug("Phase1L1TJetCalibrator")
                                         << candidate.phi() << "\t" << candidate.pileup() << std::endl;
      LogDebug("Phase1L1TJetCalibrator") << "CALIBRATION " << std::endl;
      LogDebug("Phase1L1TJetCalibrator") << "\t Using eta - pt - factor " << *etaBin << " - " << *ptBin << " - "
                                         << l1tCalibrationFactor LogDebug("Phase1L1TJetCalibrator") << std::endl;
      LogDebug("Phase1L1TJetCalibrator") << "POST-CALIBRATION " << std::endl;
      LogDebug("Phase1L1TJetCalibrator") << "\t Jet properties (pt, eta, phi, pile-up): " << newCandidate->pt() << "\t"
                                         << newCandidate->eta() << "\t" << newCandidate->phi() << "\t"
                                         << newCandidate->pileup() << std::endl;
    }
#endif
  }

  // finally, sort the collection by pt
  std::sort(calibratedCollectionPtr->begin(),
            calibratedCollectionPtr->end(),
            [](const reco::CaloJet& jet1, const reco::CaloJet& jet2) { return jet1.pt() > jet2.pt(); });

  iEvent.put(std::move(calibratedCollectionPtr), outputCollectionName_);
}

void Phase1L1TJetCalibrator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputCollectionTag",
                          edm::InputTag("l1tPhase1JetProducer", "UncalibratedPhase1L1TJetFromPfCandidates"));
  desc.add<std::vector<double>>("absEtaBinning");
  std::vector<edm::ParameterSet> vDefaults;
  edm::ParameterSetDescription validator;
  validator.add<double>("etaMax");
  validator.add<double>("etaMin");
  validator.add<std::vector<double>>("l1tCalibrationFactors");
  validator.add<std::vector<double>>("l1tPtBins");
  desc.addVPSet("calibration", validator, vDefaults);
  desc.add<std::string>("outputCollectionName", "Phase1L1TJetFromPfCandidates");
  descriptions.add("Phase1L1TJetCalibrator", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase1L1TJetCalibrator);
