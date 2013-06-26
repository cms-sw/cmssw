#ifndef HLTrigger_Muon_Test_HLTMuonTurnOnAnalyzer_H
#define HLTrigger_Muon_Test_HLTMuonTurnOnAnalyzer_H

/** \class HLTMuonTurnOnAnalyzer
 *  Get L1/HLT turn on curves
 *
 *  \author J. Alcaraz
 */

// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>

class TFile;
class TH1F;

class HLTMuonTurnOnAnalyzer: public edm::EDAnalyzer {
public:
  /// Constructor
  HLTMuonTurnOnAnalyzer(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~HLTMuonTurnOnAnalyzer();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  virtual void beginJob() ;
  virtual void endJob() ;

private:
  // Input from cfg file
  edm::InputTag theGenLabel;
  bool useMuonFromGenerator;
  edm::InputTag theL1CollectionLabel;
  std::vector<edm::InputTag> theHLTCollectionLabels;
  double theReferenceThreshold;
  double thePtMin;
  double thePtMax;
  unsigned int theNbins;
  std::string theRootFileName;

  // The output Root file
  TFile* theFile;

  // Histograms
  TH1F* hL1eff;
  std::vector <TH1F*> hHLTeff;
  TH1F* hL1nor;
  std::vector <TH1F*> hHLTnor;

  // Counter of events (weighted in general)
  double theNumberOfEvents;
  
};
#endif
