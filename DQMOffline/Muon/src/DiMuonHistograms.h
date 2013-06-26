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
  edm::InputTag bsTag;
  edm::InputTag vertexTag;

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

  //Defining the relevant invariant mass regions
  double LowMassMin;
  double LowMassMax;
  double HighMassMin;
  double HighMassMax;
  
  std::vector<MonitorElement*> GlbGlbMuon_LM;
  std::vector<MonitorElement*> GlbGlbMuon_HM;
  std::vector<MonitorElement*> StaTrkMuon_LM;
  std::vector<MonitorElement*> StaTrkMuon_HM;
  std::vector<MonitorElement*> TrkTrkMuon_LM;
  std::vector<MonitorElement*> TrkTrkMuon_HM;

  std::vector<MonitorElement*> TightTightMuon;
  std::vector<MonitorElement*> SoftSoftMuon;
  
};
#endif 

