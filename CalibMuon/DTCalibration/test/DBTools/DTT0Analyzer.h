#ifndef DTT0Analyzer_H
#define DTT0Analyzer_H

/** \class DTT0Analyzer
 *  Plot the t0 from the DB
 *
 *  \author S. Bolognesi - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include <string>
#include <fstream>
#include <vector>

class DTT0;
class TFile;
class TH1D;

class DTT0Analyzer : public edm::EDAnalyzer {
public:
  /// Constructor
  DTT0Analyzer(const edm::ParameterSet& pset);

  /// Destructor
  ~DTT0Analyzer() override;

  /// Operations
  //Read the DTGeometry and the t0 DB
  void beginRun(const edm::Run&, const edm::EventSetup& setup) override;
  void analyze(const edm::Event& event, const edm::EventSetup& setup) override {}
  //Do the real work
  void endJob() override;

protected:

private:
 std::string getHistoName(const DTLayerId& lId) const;

  //The DTGeometry
  edm::ESHandle<DTGeometry> dtGeom;

  // The file which will contain the histos
  TFile *theFile;

  //The t0 map
  const DTT0 *tZeroMap;
 
  // Map of the t0 and sigma histos by layer
  std::map<DTLayerId, TH1D*> theMeanHistoMap;
  std::map<DTLayerId, TH1D*> theSigmaHistoMap;

};
#endif

