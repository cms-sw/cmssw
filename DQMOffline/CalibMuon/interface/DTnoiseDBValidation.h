#ifndef DTnoiseDBValidation_H
#define DTnoiseDBValidation_H

/** \class DTnoiseDBValidation
 *  Plot the noise from the DB comparaison
 *
 *  \author G. Mila - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include <string>
#include <vector>
#include <map>

class DTGeometry;
class DTChamberId;
class DTStatusFlag;
class TFile;

class DTnoiseDBValidation : public edm::EDAnalyzer {
public:
  /// Constructor
  DTnoiseDBValidation(const edm::ParameterSet& pset);

    /// Destructor
  ~DTnoiseDBValidation() override;

  /// Operations
  void beginRun(const edm::Run& run, const edm::EventSetup& setup) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void endJob() override;
  void analyze(const edm::Event& event, const edm::EventSetup& setup) override {}
 
protected:

private:
  void bookHisto(const DTChamberId&);

  DQMStore* dbe_;
  // The DB label
  std::string labelDBRef_;
  std::string labelDB_;
  std::string diffTestName_,wheelTestName_,stationTestName_,
              sectorTestName_,layerTestName_;

  bool outputMEsInRootFile_; 
  std::string outputFileName_;

  // The DTGeometry
  edm::ESHandle<DTGeometry> dtGeom_;

  // The noise map
  const DTStatusFlag *noiseMap_;
  const DTStatusFlag *noiseRefMap_;
 
  //the total number of noisy cell
  int noisyCellsRef_;
  int noisyCellsValid_;
  // the histos
  MonitorElement * diffHisto_;
  MonitorElement * wheelHisto_;
  MonitorElement * stationHisto_;
  MonitorElement * sectorHisto_;
  MonitorElement * layerHisto_;
  std::map<DTChamberId, MonitorElement*> noiseHistoMap_;

};
#endif

