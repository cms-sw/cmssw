#ifndef DTt0DBValidation_H
#define DTt0DBValidation_H

/** \class DTt0DBValidation
 *  Plot the t0 from the DB
 *
 *  $Date: 2007/03/28 17:19:28 $
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
  void beginJob(const edm::EventSetup& setup);
  void analyze(const edm::Event& event, const edm::EventSetup& setup) {}
  //Do the real work
  void endJob();
  void bookHistos(DTLayerId lId, int firstWire, int lastWire);

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
 
  // Map of the t0 and sigma by wire
  std::map<DTWireId, std::vector<float> > t0RefMap;
  std::map<DTWireId, std::vector<float> > t0Map;

  // Map of the t0 difference histos by layer
  std::map<DTLayerId, MonitorElement* > t0DiffHistos;
};
#endif
