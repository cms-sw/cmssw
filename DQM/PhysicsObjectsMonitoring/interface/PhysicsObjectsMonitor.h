#ifndef DQM_PhysicsObjectMonitoring_PhysicsObjectsMonitor_H
#define DQM_PhysicsObjectMonitoring_PhysicsObjectsMonitor_H

/** \class PhysicsObjectsMonitor
 *  For now: Analyzer of StandAlone muon tracks
 *  Later: Add other detectors and more Reco
 * 
 *  $Date: 2006/10/12 07:24:51 $
 *  $Revision: 1.1 $
 *  \author M. Mulders - CERN <martijn.mulders@cern.ch>
 *  Based on STAMuonAnalyzer by R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


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
  bool saveRootFile;
  DaqMonitorBEInterface * dbe;


  std::string theSTAMuonLabel;
  std::string theSeedCollectionLabel;

  // Histograms Simulation
  MonitorElement  *hPres;
  MonitorElement  *h1_Pres;

  // Histograms MTCC data
  MonitorElement  *charge;
  MonitorElement  *ptot;
  MonitorElement  *pt;
  MonitorElement  *px;
  MonitorElement  *py;
  MonitorElement  *pz;
  MonitorElement  *Nmuon;
  MonitorElement  *Nrechits;
  MonitorElement  *NDThits;
  MonitorElement  *NCSChits;
  MonitorElement  *DTvsCSC;

  // Counters
  int numberOfSimTracks;
  int numberOfRecTracks;

  std::string theDataType;
  
};
#endif

