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
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  edm::EDGetTokenT<std::vector<reco::CaloJet>> inputCollectionTag_;
  std::vector<double> absEtaBinning_;
  size_t nBinsEta_;
  std::vector<edm::ParameterSet> calibration_;
  std::string outputCollectionName_;

  std::vector<std::vector<double>> _jetCalibrationFactorsBinnedInEta;
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
    _jetCalibrationFactorsBinnedInEta.emplace_back(pset.getParameter<std::vector<double>>("l1tCalibrationFactors"));
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

  std::unique_ptr<std::vector<reco::CaloJet>> calibratedCollectionPtr(new std::vector<reco::CaloJet>());

  // for each candidate:
  // 1 get pt and eta
  // 2 run a dicotomic search on the eta vector to find the eta index
  // 3 fetch the corresponding calibration elements
  // 4 run a dicotomic search on the pt binning to find the pt index
  // 5 fetch the calibration factor
  // 6 update the candidate pt by applying the calibration factor
  // 7 store calibrated candidate in a new collection
  for (const auto& candidate : *inputCollectionHandle) {
    // 1
    float pt = candidate.pt();
    float eta = candidate.eta();

    //2
    auto etaBin = upper_bound(absEtaBinning_.begin(), absEtaBinning_.end(), fabs(eta));
    int etaIndex = etaBin - absEtaBinning_.begin() - 1;

    //3
    const std::vector<double>& l1tPtBins = _jetCalibrationFactorsPtBins[etaIndex];
    const std::vector<double>& l1tCalibrationFactors = _jetCalibrationFactorsBinnedInEta[etaIndex];

    //4
    auto ptBin = upper_bound(l1tPtBins.begin(), l1tPtBins.end(), pt);
    int ptBinIndex = ptBin - l1tPtBins.begin() - 1;

    //5
    const double& l1tCalibrationFactor = l1tCalibrationFactors[ptBinIndex];

    //6
    reco::Candidate::PolarLorentzVector candidateP4(candidate.polarP4());
    reco::CaloJet* newCandidate = candidate.clone();
    candidateP4.SetPt(candidateP4.pt() * l1tCalibrationFactor);
    newCandidate->setP4(candidateP4);

    //7
    calibratedCollectionPtr->emplace_back(*newCandidate);
    // clean up

#ifdef DEBUG
    if (newCandidate->pt() < 0) {
      std::cout << "######################" << std::endl;
      std::cout << "PRE-CALIBRATION " << std::endl;
      std::cout << "\t Jet properties (pt, eta, phi, pile-up): " << candidate.pt() << "\t" << candidate.eta() << "\t"
                << candidate.phi() << "\t" << candidate.pileup() << std::endl;
      std::cout << "CALIBRATION " << std::endl;
      std::cout << "\t Using eta - pt - factor " << *etaBin << " - " << *ptBin << " - " << l1tCalibrationFactor
                << std::endl;
      std::cout << "POST-CALIBRATION " << std::endl;
      std::cout << "\t Jet properties (pt, eta, phi, pile-up): " << newCandidate->pt() << "\t" << newCandidate->eta()
                << "\t" << newCandidate->phi() << "\t" << newCandidate->pileup() << std::endl;
    }
#endif

    delete newCandidate;
  }

  // finally, sort the collection by pt
  std::sort(calibratedCollectionPtr->begin(),
            calibratedCollectionPtr->end(),
            [](const reco::CaloJet& jet1, const reco::CaloJet& jet2) { return jet1.pt() > jet2.pt(); });

  iEvent.put(std::move(calibratedCollectionPtr), outputCollectionName_);
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void Phase1L1TJetCalibrator::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void Phase1L1TJetCalibrator::endStream() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void Phase1L1TJetCalibrator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase1L1TJetCalibrator);
