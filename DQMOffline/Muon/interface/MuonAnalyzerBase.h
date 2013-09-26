#ifndef MuonAnalyzerBase_H
#define MuonAnalyzerBase_H


/** \class MuonAnalyzerBase
 *
 *  base class for all DQM monitor sources
 *
 *  \author G. Mila - INFN Torino
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

class MuonAnalyzerBase : public edm::EDAnalyzer {
 public:

  /// Constructor
  MuonAnalyzerBase(MuonServiceProxy *theServ):theService(theServ){}
  
  /// Destructor
  virtual ~MuonAnalyzerBase() {}
  
  /// Inizialize parameters for histo binning
  virtual void beginJob(DQMStore* dbe)= 0;

  /// Get the analysis of the muon properties
  void analyze(const edm::Event&, const edm::EventSetup&, reco::Muon& recoMuon){}

  /// Get the analysis of the muon track properties
  void analyze(const edm::Event&, const edm::EventSetup&, reco::Track& recoTrack){}
  void analyze(const edm::Event&, const edm::EventSetup&){};
  
  MuonServiceProxy* service() {return theService;}

 private:
  // ----------member data ---------------------------
  MuonServiceProxy *theService;
};
#endif  
