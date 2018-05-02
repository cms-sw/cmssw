#ifndef DQMOFFLINE_TRIGGER_DIDISPSTAMUONMONITOR_H
#define DQMOFFLINE_TRIGGER_DIDISPSTAMUONMONITOR_H


#include <string>
#include <vector>
#include <map>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

//DataFormats
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class GenericTriggerEventFlag;


//
// class declaration
//

class DiDispStaMuonMonitor : public DQMEDAnalyzer 
{
public:
  DiDispStaMuonMonitor( const edm::ParameterSet& );
  ~DiDispStaMuonMonitor() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);

  struct DiDispStaMuonME {
    MonitorElement* numerator;
    MonitorElement* denominator;
  };

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void bookME(DQMStore::IBooker &, DiDispStaMuonME& me, const std::string& histname, const std::string& histtitle, int nbins, double xmin, double xmax);
  void bookME(DQMStore::IBooker &, DiDispStaMuonME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX);
  void bookME(DQMStore::IBooker &, DiDispStaMuonME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax, bool bookDen);
  void bookME(DQMStore::IBooker &, DiDispStaMuonME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, DiDispStaMuonME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY);
  void setTitle(DiDispStaMuonME& me, const std::string& titleX, const std::string& titleY, bool bookDen);

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:  
  struct MEbinning {
    unsigned int nbins;
    double xmin;
    double xmax;
  };

  static MEbinning getHistoPSet    (const edm::ParameterSet & pset);
  static MEbinning getHistoLSPSet  (const edm::ParameterSet & pset);

  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::TrackCollection>        muonToken_;

  std::vector<double> muonPt_variable_binning_;
  MEbinning           muonPt_binning_;
  MEbinning           muonEta_binning_;
  MEbinning           muonPhi_binning_;
  MEbinning           muonDxy_binning_;
  MEbinning           ls_binning_;

  DiDispStaMuonME muonPtME_;
  DiDispStaMuonME muonPtNoDxyCutME_;
  DiDispStaMuonME muonPtME_variableBinning_;
  DiDispStaMuonME muonPtVsLS_;
  DiDispStaMuonME muonEtaME_;
  DiDispStaMuonME muonPhiME_;
  DiDispStaMuonME muonDxyME_;
  DiDispStaMuonME subMuonPtME_;
  DiDispStaMuonME subMuonPtME_variableBinning_;
  DiDispStaMuonME subMuonEtaME_;
  DiDispStaMuonME subMuonPhiME_;
  DiDispStaMuonME subMuonDxyME_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::Track,true>        muonSelectionGeneral_;
  StringCutObjectSelector<reco::Track,true>        muonSelectionPt_;
  StringCutObjectSelector<reco::Track,true>        muonSelectionDxy_;
  unsigned int nmuons_;

};

#endif //DQMOFFLINE_TRIGGER_DIDISPSTAMUONMONITOR_H
