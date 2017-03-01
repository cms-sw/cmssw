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
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"


class DiMuonHistograms : public DQMEDAnalyzer {
 public:
  /* Constructor */ 
  DiMuonHistograms(const edm::ParameterSet& pset);
  
  /* Destructor */ 
  virtual ~DiMuonHistograms() ;
  
  /* Operations */ 
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  
 private:
  MuonServiceProxy* theService;
  edm::ParameterSet parameters;
  
  // Switch for verbosity
  std::string metname;
  
  //histo binning parameters
  int etaBin;
  int etaBBin;
  int etaEBin;
  int etaOvlpBin;

  //Defining relevant eta regions
  std::string EtaName[3];

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
  
  MonitorElement* test; // my test

  // Labels used
  edm::EDGetTokenT<edm::View<reco::Muon> >   theMuonCollectionLabel_;
  edm::EDGetTokenT<reco::VertexCollection> theVertexLabel_;
  edm::EDGetTokenT<reco::BeamSpot>         theBeamSpotLabel_;

  std::string theFolder;

  int nTightTight;
  int nGlbGlb;

};
#endif 

