#ifndef DQM_PhysicsObjectMonitoring_PhysicsObjectsMonitor_H
#define DQM_PhysicsObjectMonitoring_PhysicsObjectsMonitor_H

/** \class PhysicsObjectsMonitor
 *  For now: Analyzer of StandAlone muon tracks
 *  Later: Add other detectors and more Reco
 * 
 *  $Date: 2006/07/18 10:18:22 $
 *  $Revision: 1.1 $
 *  \author M. Mulders - CERN <martijn.mulders@cern.ch>
 *  Based on STAMuonAnalyzer by R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TFile;
class TH1F;
class TH2F;

class PhysicsObjectsMonitor: public edm::EDAnalyzer {
public:
  /// Constructor
  PhysicsObjectsMonitor(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~PhysicsObjectsMonitor();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  virtual void beginJob(const edm::EventSetup& eventSetup) ;
  virtual void endJob() ;
protected:

private:
  std::string theRootFileName;
  TFile* theFile;

  std::string theSTAMuonLabel;
  std::string theSeedCollectionLabel;

  // Histograms Simulation
  TH1F *hPres;
  TH1F *h1_Pres;

  // Histograms MTCC data
  TH1F *charge;
  TH1F *ptot;
  TH1F *pt;
  TH1F *px;
  TH1F *py;
  TH1F *pz;
  TH1F *Nmuon;
  TH1F *Nrechits;
  TH1F *NDThits;
  TH1F *NCSChits;
  TH2F *DTvsCSC;

  // Counters
  int numberOfSimTracks;
  int numberOfRecTracks;

  std::string theDataType;
  
};
#endif

