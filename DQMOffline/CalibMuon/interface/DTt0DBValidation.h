#ifndef DTt0DBValidation_H
#define DTt0DBValidation_H

/** \class DTt0DBValidation
 *  Plot the t0 from the DB
 *
 *  $Date: 2009/02/16 14:04:02 $
 *  $Revision: 1.3 $
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

class DTT0;
class TFile;

class DTt0DBValidation : public edm::EDAnalyzer {
public:
  /// Constructor
  DTt0DBValidation(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTt0DBValidation();

  /// Operations
  //Read the DTGeometry and the t0 DB
  void beginRun(const edm::Run& run, const edm::EventSetup& setup);
  void analyze(const edm::Event& event, const edm::EventSetup& setup) {}
  //Do the real work
  void endJob();
  void bookHistos(DTLayerId lId, int firstWire, int lastWire);
  void bookHistos(int wheel);

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

  // The t0 map
  const DTT0 *tZeroMap;
  const DTT0 *tZeroRefMap;
 
  // Map of the t0 and sigma per wire
  std::map<DTWireId, std::vector<float> > t0RefMap;
  std::map<DTWireId, std::vector<float> > t0Map;

  // Map of the t0 difference histos per layer
  std::map<DTLayerId, MonitorElement* > t0DiffHistos;

  // summary histos
  std::map<int, MonitorElement* > wheelSummary;
};
#endif
