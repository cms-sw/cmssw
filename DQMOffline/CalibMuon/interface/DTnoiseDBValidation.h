#ifndef DTnoiseDBValidation_H
#define DTnoiseDBValidation_H

/** \class DTnoiseDBValidation
 *  Plot the noise from the DB comparaison
 *
 *  $Date: 2008/09/24 14:49:18 $
 *  $Revision: 1.1 $
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

#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include <string>
#include <fstream>
#include <vector>

class DTStatusFlag;
class TFile;

class DTnoiseDBValidation : public edm::EDAnalyzer {
public:
  /// Constructor
  DTnoiseDBValidation(const edm::ParameterSet& pset);

    /// Destructor
  virtual ~DTnoiseDBValidation();

  /// Operations
  //Read the DTGeometry and the t0 DB
  void beginJob();
  void beginRun(const edm::Run& run, const edm::EventSetup& setup);

  void analyze(const edm::Event& event, const edm::EventSetup& setup) {}
  //Do the real work
  void endJob();
 
protected:

private:

  DQMStore* dbe;
  edm::ParameterSet parameters;
  // Switch for verbosity
  std::string metname;
  // The DB label
  std::string labelDBRef;
  std::string labelDB;
  // The file which will contain the difference plot
  std::string outputFileName;

  // The DTGeometry
  edm::ESHandle<DTGeometry> dtGeom;

  // The noise map
  const DTStatusFlag *noiseMap;
  const DTStatusFlag *noiseRefMap;
 
  //the total number of noisy cell
  int noisyCells_Ref;
  int noisyCells_toTest;
  // the histos
  MonitorElement * diffHisto;
  MonitorElement * wheelHisto;
  MonitorElement * stationHisto;
  MonitorElement * sectorHisto;
  MonitorElement * layerHisto;

};
#endif

