#ifndef DQMOffline_Muon_MuonTiming_H
#define DQMOffline_Muon_MuonTiming_H

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
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

class MuonTiming : public DQMEDAnalyzer {
public:
  /// Constructor
  MuonTiming(const edm::ParameterSet &);

  /// Destructor
  ~MuonTiming() override;

  /// Inizialize parameters for histo binning
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<reco::Muon> > theMuonCollectionLabel_;
  // Switch for verbosity
  std::string metname_;

  //histo binning parameters
  int tnbins_;
  int tnbinsrpc_;
  int terrnbins_;
  int terrnbinsrpc_;
  int ndofnbins_;
  int ptnbins_;
  int etanbins_;
  double tmax_, tmin_;
  double tmaxrpc_, tminrpc_;
  double terrmax_, terrmin_;
  double terrmaxrpc_, terrminrpc_;
  double ndofmax_, ndofmin_;
  double ptmax_, ptmin_;
  double etamax_, etamin_;
  double etaBarrelMin_, etaBarrelMax_, etaEndcapMin_, etaEndcapMax_, etaOverlapMin_, etaOverlapMax_;

  std::string theFolder_;
  std::vector<std::string> EtaName_, ObjectName_;
  enum eta_ { overlap, barrel, endcap };
  enum object_ { sta, glb };
  //the histos
  /*
  std::vector<std::vector<MonitorElement*>> timeNDof_;
  std::vector<std::vector<MonitorElement*>> timeAtIpInOut_;
  std::vector<std::vector<MonitorElement*>> timeAtIpInOutRPC_;
  std::vector<std::vector<MonitorElement*>> timeAtIpInOutErr_;
  std::vector<std::vector<MonitorElement*>> timeAtIpInOutErrRPC_;
  std::vector<MonitorElement*> etaptVeto_;
  std::vector<MonitorElement*> etaVeto_;
  std::vector<MonitorElement*> ptVeto_;
  std::vector<MonitorElement*> yields_;
  */
  std::array<std::array<MonitorElement *, 1>, 3> timeNDof_;
  std::array<std::array<MonitorElement *, 1>, 3> timeAtIpInOut_;
  std::array<std::array<MonitorElement *, 1>, 3> timeAtIpInOutRPC_;
  std::array<std::array<MonitorElement *, 1>, 3> timeAtIpInOutErr_;
  std::array<std::array<MonitorElement *, 1>, 3> timeAtIpInOutErrRPC_;
  std::array<MonitorElement *, 1> etaptVeto_;
  std::array<MonitorElement *, 1> etaVeto_;
  std::array<MonitorElement *, 1> ptVeto_;
  std::array<MonitorElement *, 1> yields_;
};
#endif
