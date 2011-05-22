#ifndef DIMUONHISTOGRAMS_H
#define DIMUONHISTOGRAMS_H

/**   Class DiMuonHistograms
 *  
 *    DQM monitoring for dimuon mass
 *    
 *    Author:  S.Folgueras, U. Oviedo
 */

/* Base Class Headers */
#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/Muon/src/MuonAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

class DiMuonHistograms : public MuonAnalyzerBase {
  
 public:
  /* Constructor */ 
  DiMuonHistograms(const edm::ParameterSet& pset, MuonServiceProxy *theService);
  
  /* Destructor */ 
  virtual ~DiMuonHistograms() ;

  /* Operations */ 
  void beginJob (DQMStore *dbe);
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);
  
 protected:
  edm::ParameterSet parameters;
  
  // Switch for verbosity
  std::string metname;
  
  // STA Label
  edm::InputTag theSTACollectionLabel;
  edm::InputTag theMuonCollectionLabel;

  //histo binning parameters
  int etaBin;
  int etaBBin;
  int etaEBin;
  int etaOvlpBin;

  //Defining relevant eta regions
  std::string EtaName;

  double EtaCutMin;
  double EtaCutMax;
  double etaBMin;
  double etaBMax;
  double etaECMin;
  double etaECMax;
  double etaOvlpMin;
  double etaOvlpMax;

  std::vector<MonitorElement*> GlbGlbMuon;
  std::vector<MonitorElement*> GlbStaMuon;
  std::vector<MonitorElement*> GlbTrkMuon;
  std::vector<MonitorElement*> GlbGPTMuon;
  
  //  std::vector<MonitorElement*> StaGlbMuon;
  std::vector<MonitorElement*> StaStaMuon;
  std::vector<MonitorElement*> StaTrkMuon;
  std::vector<MonitorElement*> StaGPTMuon;

  //  std::vector<MonitorElement*> TrkGlbMuon;
  //  std::vector<MonitorElement*> TrkStaMuon;
  std::vector<MonitorElement*> TrkTrkMuon;
  std::vector<MonitorElement*> TrkGPTMuon;

  //  std::vector<MonitorElement*> GPTGlbMuon;
  //  std::vector<MonitorElement*> GPTStaMuon;
  //  std::vector<MonitorElement*> GPTTrkMuon;
  std::vector<MonitorElement*> GPTGPTMuon;
  
};
#endif 

