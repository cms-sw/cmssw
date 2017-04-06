#ifndef MuonTiming_H
#define MuonTiming_H

/** \class MuRecoAnalyzer
 *
 *  DQM monitoring source for muon reco track
 *
 *  \author G. Mila - INFN Torino
 */

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 

class MuonTiming : public DQMEDAnalyzer {
 public:

  /// Constructor
  MuonTiming(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~MuonTiming();

  /// Inizialize parameters for histo binning
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

 private:
  // ----------member data ---------------------------
  MuonServiceProxy *theService;
  edm::ParameterSet parameters;
  
  edm::EDGetTokenT<edm::View<reco::Muon> >   theMuonCollectionLabel_;
  // Switch for verbosity
  std::string metname;
    

  //histo binning parameters
  int tnbins;
  int tnbinsrpc;
  int terrnbins;
  int terrnbinsrpc;
  int ndofnbins;
  int ptnbins;
  int etanbins;
  double tmax, tmin;
  double tmaxrpc, tminrpc;
  double terrmax, terrmin;
  double terrmaxrpc, terrminrpc;
  double ndofmax, ndofmin;
  double ptmax, ptmin;
  double etamax, etamin;
  double etaBarrelMin, etaBarrelMax, etaEndcapMin, etaEndcapMax, etaOverlapMin, etaOverlapMax;

  std::string theFolder;
  std::vector<std::string> EtaName, ObjectName;
  enum eta {overlap, barrel, endcap};
  enum object {sta, glb};
  //the histos
  std::vector<std::vector<MonitorElement*>> timeNDof;
  std::vector<std::vector<MonitorElement*>> timeAtIpInOut;
  std::vector<std::vector<MonitorElement*>> timeAtIpInOutRPC;
  std::vector<std::vector<MonitorElement*>> timeAtIpInOutErr;
  std::vector<std::vector<MonitorElement*>> timeAtIpInOutErrRPC;
  std::vector<MonitorElement*> etaptVeto;
  std::vector<MonitorElement*> etaVeto;
  std::vector<MonitorElement*> ptVeto;
  std::vector<MonitorElement*> yields;

};
#endif
