#ifndef DQMOFFLINE_TRIGGER_NOBPTXMONITOR_H
#define DQMOFFLINE_TRIGGER_NOBPTXMONITOR_H


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
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class GenericTriggerEventFlag;


//
// class declaration
//

class NoBPTXMonitor : public DQMEDAnalyzer 
{
public:
  NoBPTXMonitor( const edm::ParameterSet& );
  ~NoBPTXMonitor() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);

  struct NoBPTXME {
    MonitorElement* numerator;
    MonitorElement* denominator;
  };

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void bookNoBPTX(DQMStore::IBooker &, NoBPTXME& me, const std::string& histname, const std::string& histtitle, int nbins, double xmin, double xmax);
  void bookNoBPTX(DQMStore::IBooker &, NoBPTXME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX);
  void bookNoBPTX(DQMStore::IBooker &, NoBPTXME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax, bool bookDen);
  void bookNoBPTX(DQMStore::IBooker &, NoBPTXME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax);
  void bookNoBPTX(DQMStore::IBooker &, NoBPTXME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY);
  void setNoBPTXTitle(NoBPTXME& me, const std::string& titleX, const std::string& titleY, bool bookDen);

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:  
  struct NoBPTXbinning {
    unsigned int nbins;
    double xmin;
    double xmax;
  };

  static NoBPTXbinning getHistoPSet    (const edm::ParameterSet & pset);
  static NoBPTXbinning getHistoLSPSet  (const edm::ParameterSet & pset);

  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::CaloJetCollection>       jetToken_;
  edm::EDGetTokenT<reco::TrackCollection>        muonToken_;

  std::vector<double> jetE_variable_binning_;
  NoBPTXbinning           jetE_binning_;
  NoBPTXbinning           jetEta_binning_;
  NoBPTXbinning           jetPhi_binning_;
  std::vector<double> muonPt_variable_binning_;
  NoBPTXbinning           muonPt_binning_;
  NoBPTXbinning           muonEta_binning_;
  NoBPTXbinning           muonPhi_binning_;
  NoBPTXbinning           ls_binning_;
  NoBPTXbinning           bx_binning_;

  NoBPTXME jetENoBPTX_;
  NoBPTXME jetENoBPTX_variableBinning_;
  NoBPTXME jetEVsLS_;
  NoBPTXME jetEVsBX_;
  NoBPTXME jetEtaNoBPTX_;
  NoBPTXME jetEtaVsLS_;
  NoBPTXME jetEtaVsBX_;
  NoBPTXME jetPhiNoBPTX_;
  NoBPTXME jetPhiVsLS_;
  NoBPTXME jetPhiVsBX_;
  NoBPTXME muonPtNoBPTX_;
  NoBPTXME muonPtNoBPTX_variableBinning_;
  NoBPTXME muonPtVsLS_;
  NoBPTXME muonPtVsBX_;
  NoBPTXME muonEtaNoBPTX_;
  NoBPTXME muonEtaVsLS_;
  NoBPTXME muonEtaVsBX_;
  NoBPTXME muonPhiNoBPTX_;
  NoBPTXME muonPhiVsLS_;
  NoBPTXME muonPhiVsBX_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::CaloJet,true   >    jetSelection_;
  StringCutObjectSelector<reco::Track,true>        muonSelection_;
  unsigned int njets_;
  unsigned int nmuons_;

};

#endif //DQMOFFLINE_TRIGGER_NOBPTXMONITOR_H
