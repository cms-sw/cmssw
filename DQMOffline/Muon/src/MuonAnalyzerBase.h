#ifndef MuonAnalyzerBase_H
#define MuonAnalyzerBase_H


/** \class MuonAnalyzerBase
 *
 *  base class for all DQM monitor sources
 *
 *  $Date: 2008/03/18 11:58:07 $
 *  $Revision: 1.1 $
 *  \author G. Mila - INFN Torino
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

class MuonAnalyzerBase {
 public:

  /// Constructor
  MuonAnalyzerBase(MuonServiceProxy *theServ){theService = theServ;}
  
  /// Destructor
  virtual ~MuonAnalyzerBase() {}
  
  /// Inizialize parameters for histo binning
  virtual void beginJob(edm::EventSetup const& iSetup, DaqMonitorBEInterface * dbe)= 0;

  /// Get the analysis of the muon properties
  void analyze(const edm::Event&, const edm::EventSetup&, reco::Muon& recoMuon){}

  /// Get the analysis of the muon track properties
  void analyze(const edm::Event&, const edm::EventSetup&, reco::Track& recoTrack){}

  MuonServiceProxy* service() {return theService;}

 private:
  // ----------member data ---------------------------
  MuonServiceProxy *theService;

};
#endif  
