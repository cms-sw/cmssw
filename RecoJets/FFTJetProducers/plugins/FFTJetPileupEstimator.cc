// -*- C++ -*-
//
// Package:    RecoJets/FFTJetProducers
// Class:      FFTJetPileupEstimator
//
/**\class FFTJetPileupEstimator FFTJetPileupEstimator.cc RecoJets/FFTJetProducers/plugins/FFTJetPileupEstimator.cc

 Description: applies calibration curve and estimates the actual pileup

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Wed Apr 20 13:52:23 CDT 2011
//
//

#include <cmath>

// Framework include files
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Data formats
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
// #include "DataFormats/Histograms/interface/MEtoEDMFormat.h"
#include "DataFormats/JetReco/interface/FFTJetPileupSummary.h"
#include "DataFormats/JetReco/interface/DiscretizedEnergyFlow.h"

#include "RecoJets/FFTJetProducers/interface/FFTJetParameterParser.h"

// Loader for the lookup tables
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetLookupTableSequenceLoader.h"

#define init_param(type, varname) varname(ps.getParameter<type>(#varname))

using namespace fftjetcms;

//
// class declaration
//
class FFTJetPileupEstimator : public edm::stream::EDProducer<> {
public:
  explicit FFTJetPileupEstimator(const edm::ParameterSet&);
  FFTJetPileupEstimator() = delete;
  FFTJetPileupEstimator(const FFTJetPileupEstimator&) = delete;
  FFTJetPileupEstimator& operator=(const FFTJetPileupEstimator&) = delete;
  ~FFTJetPileupEstimator() override;

protected:
  // methods
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  std::unique_ptr<reco::FFTJetPileupSummary> calibrateFromConfig(double uncalibrated) const;

  std::unique_ptr<reco::FFTJetPileupSummary> calibrateFromDB(double uncalibrated, const edm::EventSetup& iSetup) const;

  template <class Ptr>
  inline void checkConfig(const Ptr& ptr, const char* message) {
    if (ptr.get() == nullptr)
      throw cms::Exception("FFTJetBadConfig") << message << std::endl;
  }

  edm::InputTag inputLabel;
  edm::EDGetTokenT<reco::DiscretizedEnergyFlow> inputToken;

  std::string outputLabel;
  double cdfvalue;
  double ptToDensityFactor;
  unsigned filterNumber;
  std::vector<double> uncertaintyZones;
  std::unique_ptr<fftjet::Functor1<double, double> > calibrationCurve;
  std::unique_ptr<fftjet::Functor1<double, double> > uncertaintyCurve;

  // Alternative method to calibrate the pileup.
  // We will fetch three lookup tables from the database:
  //
  // 1. calibration curve table
  //
  // 2. uncertainty curve table
  //
  // 3. the table that will lookup the uncertainty code
  //    given the uncalibrated pileup value
  //
  // It will be assumed that all these tables will have
  // the same record and category but different names.
  //
  std::string calibTableRecord;
  std::string calibTableCategory;
  std::string uncertaintyZonesName;
  std::string calibrationCurveName;
  std::string uncertaintyCurveName;
  bool loadCalibFromDB;
};

//
// constructors and destructor
//
FFTJetPileupEstimator::FFTJetPileupEstimator(const edm::ParameterSet& ps)
    : init_param(edm::InputTag, inputLabel),
      init_param(std::string, outputLabel),
      init_param(double, cdfvalue),
      init_param(double, ptToDensityFactor),
      init_param(unsigned, filterNumber),
      init_param(std::vector<double>, uncertaintyZones),
      init_param(std::string, calibTableRecord),
      init_param(std::string, calibTableCategory),
      init_param(std::string, uncertaintyZonesName),
      init_param(std::string, calibrationCurveName),
      init_param(std::string, uncertaintyCurveName),
      init_param(bool, loadCalibFromDB) {
  calibrationCurve = fftjet_Function_parser(ps.getParameter<edm::ParameterSet>("calibrationCurve"));
  checkConfig(calibrationCurve, "bad calibration curve definition");

  uncertaintyCurve = fftjet_Function_parser(ps.getParameter<edm::ParameterSet>("uncertaintyCurve"));
  checkConfig(uncertaintyCurve, "bad uncertainty curve definition");

  inputToken = consumes<reco::DiscretizedEnergyFlow>(inputLabel);

  produces<reco::FFTJetPileupSummary>(outputLabel);
}

FFTJetPileupEstimator::~FFTJetPileupEstimator() {}

//
// member functions
//

// ------------ method called to for each event  ------------
void FFTJetPileupEstimator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::DiscretizedEnergyFlow> input;
  iEvent.getByToken(inputToken, input);

  const reco::DiscretizedEnergyFlow& h(*input);
  const unsigned nScales = h.nEtaBins();
  const unsigned nCdfvalues = h.nPhiBins();

  const unsigned fixedCdfvalueBin = static_cast<unsigned>(std::floor(cdfvalue * nCdfvalues));
  if (fixedCdfvalueBin >= nCdfvalues) {
    throw cms::Exception("FFTJetBadConfig") << "Bad cdf value" << std::endl;
  }
  if (filterNumber >= nScales) {
    throw cms::Exception("FFTJetBadConfig") << "Bad filter number" << std::endl;
  }

  // Simple fixed-point pile-up estimate
  const double curve = h.data()[filterNumber * nCdfvalues + fixedCdfvalueBin];

  std::unique_ptr<reco::FFTJetPileupSummary> summary;
  if (loadCalibFromDB)
    summary = calibrateFromDB(curve, iSetup);
  else
    summary = calibrateFromConfig(curve);
  iEvent.put(std::move(summary), outputLabel);
}

std::unique_ptr<reco::FFTJetPileupSummary> FFTJetPileupEstimator::calibrateFromConfig(const double curve) const {
  const double pileupRho = ptToDensityFactor * (*calibrationCurve)(curve);
  const double rhoUncert = ptToDensityFactor * (*uncertaintyCurve)(curve);

  // Determine the uncertainty zone of the estimate. The "curve"
  // has to be above or equal to uncertaintyZones[i]  but below
  // uncertaintyZones[i + 1] (the second condition is also satisfied
  // by i == uncertaintyZones.size() - 1). Of course, it is assumed
  // that the vector of zones is configured appropriately -- the zone
  // boundaries must be presented in the increasing order.
  int uncertaintyCode = -1;
  if (!uncertaintyZones.empty()) {
    const unsigned nZones = uncertaintyZones.size();
    for (unsigned i = 0; i < nZones; ++i)
      if (curve >= uncertaintyZones[i]) {
        if (i == nZones - 1U) {
          uncertaintyCode = i;
          break;
        } else if (curve < uncertaintyZones[i + 1]) {
          uncertaintyCode = i;
          break;
        }
      }
  }

  return std::make_unique<reco::FFTJetPileupSummary>(curve, pileupRho, rhoUncert, uncertaintyCode);
}

std::unique_ptr<reco::FFTJetPileupSummary> FFTJetPileupEstimator::calibrateFromDB(const double curve,
                                                                                  const edm::EventSetup& iSetup) const {
  edm::ESHandle<FFTJetLookupTableSequence> h;
  StaticFFTJetLookupTableSequenceLoader::instance().load(iSetup, calibTableRecord, h);
  std::shared_ptr<npstat::StorableMultivariateFunctor> uz = (*h)[calibTableCategory][uncertaintyZonesName];
  std::shared_ptr<npstat::StorableMultivariateFunctor> cc = (*h)[calibTableCategory][calibrationCurveName];
  std::shared_ptr<npstat::StorableMultivariateFunctor> uc = (*h)[calibTableCategory][uncertaintyCurveName];

  const double pileupRho = ptToDensityFactor * (*cc)(&curve, 1U);
  const double rhoUncert = ptToDensityFactor * (*uc)(&curve, 1U);
  const int uncertaintyCode = round((*uz)(&curve, 1U));

  return std::make_unique<reco::FFTJetPileupSummary>(curve, pileupRho, rhoUncert, uncertaintyCode);
}

//define this as a plug-in
DEFINE_FWK_MODULE(FFTJetPileupEstimator);
