#ifndef Calibration_EcalCalibAlgos_miscalibExample_h
#define Calibration_EcalCalibAlgos_miscalibExample_h

/**\class miscalibExample

 Description: Analyzer to fetch collection of objects from event and make simple plots

 Implementation:
     \\\author: Lorenzo Agostino, September 2006
*/
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include <string>
#include "TH1.h"
#include "TFile.h"
//
// class declaration
//

class miscalibExample : public edm::one::EDAnalyzer<> {
public:
  explicit miscalibExample(const edm::ParameterSet&);
  ~miscalibExample() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override;
  void endJob() override;

private:
  // ----------member data ---------------------------
  const std::string rootfile_;
  const std::string correctedHybridSuperClusterProducer_;
  const std::string correctedHybridSuperClusterCollection_;
  int read_events;
  const edm::EDGetTokenT<reco::SuperClusterCollection> correctedHybridSuperClusterToken_;

  TH1F* scEnergy;
};

#endif
