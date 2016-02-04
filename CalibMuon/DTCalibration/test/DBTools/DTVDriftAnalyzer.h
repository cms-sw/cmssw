#ifndef DTVDriftAnalyzer_H
#define DTVDriftAnalyzer_H

/** \class DTVDriftAnalyzer
 *  Plot the vdrift from the DB
 *
 *  $Date: 2008/10/03 08:53:22 $
 *  $Revision: 1.3 $
 *  \author S. Bolognesi - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include <string>
#include <fstream>
#include <vector>

class DTMtime;
class TFile;
class TH1D;

class DTVDriftAnalyzer : public edm::EDAnalyzer {
public:
  /// Constructor
  DTVDriftAnalyzer(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTVDriftAnalyzer();

  /// Operations
  //Read the DTGeometry and the vdrift DB
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& setup );
  void analyze(const edm::Event& event, const edm::EventSetup& setup) {}
  //Do the real work
  void endJob();

protected:

private:
  std::string getHistoName(const DTWireId& lId) const;
  std::string getDistribName(const DTWireId& wId) const;

  // The file which will contain the histos
  TFile *theFile;

  //The t0 map
  const DTMtime *mTimeMap;
  
  // Map of the vdrift, reso histos by wheel/sector/SL
  std::map<std::pair<int,int>, TH1D*> theVDriftHistoMap;
  std::map<std::pair<int,int>, TH1D*> theResoHistoMap;
 // Map of the vdrift, reso distributions by wheel/station/SL
  std::map<std::vector<int>, TH1D*> theVDriftDistribMap;
  std::map<std::vector<int>, TH1D*> theResoDistribMap;

};
#endif

