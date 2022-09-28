#ifndef HLTrigger_Muon_Test_HLTMuonRateAnalyzer_H
#define HLTrigger_Muon_Test_HLTMuonRateAnalyzer_H

/** \class HLTMuonRateAnalyzer
 *  Get L1/HLT efficiency/rate plots
 *
 *  \author J. Alcaraz
 */

// Base Class Headers
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

#include <vector>

class TFile;
class TH1F;

class HLTMuonRateAnalyzer : public edm::one::EDAnalyzer<> {
public:
  /// Constructor
  HLTMuonRateAnalyzer(const edm::ParameterSet& pset);

  /// Destructor
  ~HLTMuonRateAnalyzer() override;

  // Operations

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;

  void beginJob() override;
  void endJob() override;

private:
  // Input from cfg file
  edm::InputTag theGenLabel;
  edm::InputTag theL1CollectionLabel;
  std::vector<edm::InputTag> theHLTCollectionLabels;
  edm::EDGetTokenT<edm::HepMCProduct> theGenToken;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> theL1CollectionToken;
  std::vector<edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> > theHLTCollectionTokens;
  double theL1ReferenceThreshold;
  std::vector<double> theNSigmas;
  unsigned int theNumberOfObjects;
  double theCrossSection;
  double theLuminosity;
  double thePtMin;
  double thePtMax;
  unsigned int theNbins;
  std::string theRootFileName;

  // The output Root file
  TFile* theFile;

  // Histograms
  TH1F* hL1eff;
  TH1F* hL1rate;
  std::vector<TH1F*> hHLTeff;
  std::vector<TH1F*> hHLTrate;

  // Counter of events (weighted in general)
  double theNumberOfEvents;
};
#endif
