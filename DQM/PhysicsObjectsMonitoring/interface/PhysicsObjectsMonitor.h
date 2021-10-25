#ifndef DQM_PhysicsObjectMonitoring_PhysicsObjectsMonitor_H
#define DQM_PhysicsObjectMonitoring_PhysicsObjectsMonitor_H

/** \class PhysicsObjectsMonitor
 *  For now: Analyzer of StandAlone muon tracks
 *  Later: Add other detectors and more Reco
 *
 *  \author M. Mulders - CERN <martijn.mulders@cern.ch>
 *  Based on STAMuonAnalyzer by R. Bellan - INFN Torino
 *<riccardo.bellan@cern.ch>
 */

// Base Class Headers
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class TFile;
class TH1F;
class TH2F;
class MagneticField;
class IdealMagneticFieldRecord;

class PhysicsObjectsMonitor : public DQMEDAnalyzer {
public:
  /// Constructor
  PhysicsObjectsMonitor(const edm::ParameterSet &pset);
  /// Destructor
  ~PhysicsObjectsMonitor() override;
  // Operations
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &event, const edm::EventSetup &eventSetup) override;

private:
  std::string theSTAMuonLabel;
  std::string theSeedCollectionLabel;

  // Histograms Simulation
  MonitorElement *hPres;
  MonitorElement *h1_Pres;

  // Histograms MTCC data
  MonitorElement *charge;
  MonitorElement *ptot;
  MonitorElement *pt;
  MonitorElement *px;
  MonitorElement *py;
  MonitorElement *pz;
  MonitorElement *Nmuon;
  MonitorElement *Nrechits;
  MonitorElement *NDThits;
  MonitorElement *NCSChits;
  MonitorElement *DTvsCSC;
  MonitorElement *DTvsRPC;
  MonitorElement *CSCvsRPC;
  MonitorElement *NRPChits;

  std::string theDataType;

  // define Token(-s)
  edm::EDGetTokenT<reco::TrackCollection> theSTAMuonToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFiledToken_;
};
#endif
