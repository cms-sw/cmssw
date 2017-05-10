#ifndef DTt0DBValidation_H
#define DTt0DBValidation_H

/** \class DTt0DBValidation
 *  Plot the t0 from the DB
 *
 *  \author G. Mila - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
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

class DTt0DBValidation : public DQMEDAnalyzer {
public:
  /// Constructor
  DTt0DBValidation(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTt0DBValidation();

  /// Operations
  //Read the DTGeometry and the t0 DB
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  virtual void endStream(void) override;

private:
  void bookHistos(DQMStore::IBooker &, DTLayerId lId, int firstWire, int lastWire);
  void bookHistos(DQMStore::IBooker &, int wheel);

  // Switch for verbosity
  std::string metname_;
  // The DB label
  std::string labelDBRef_;
  std::string labelDB_;

  std::string t0TestName_;

  // The DTGeometry
  edm::ESHandle<DTGeometry> dtGeom_;

  // The t0 map
  const DTT0 *tZeroMap_;
  const DTT0 *tZeroRefMap_;
 
  // Map of the t0 and sigma per wire
  std::map<DTWireId, std::vector<float> > t0RefMap_;
  std::map<DTWireId, std::vector<float> > t0Map_;

  // Map of the t0 difference histos per layer
  std::map<DTLayerId, MonitorElement* > t0DiffHistos_;

  // Summary histos
  std::map<int, MonitorElement* > wheelSummary_;
};
#endif
