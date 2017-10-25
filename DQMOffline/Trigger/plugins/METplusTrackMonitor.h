#ifndef DQMOFFLINE_TRIGGER_METPLUSTRACKMONITOR_H
#define DQMOFFLINE_TRIGGER_METPLUSTRACKMONITOR_H

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

// DataFormats
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"


class GenericTriggerEventFlag;

//
// class declaration
//

class METplusTrackMonitor : public DQMEDAnalyzer
{
public:

  struct MEbinning {
    unsigned int nbins;
    double xmin;
    double xmax;
  };

  struct METplusTrackME {
    MonitorElement * numerator;
    MonitorElement * denominator;
  };

  METplusTrackMonitor( const edm::ParameterSet& );
  ~METplusTrackMonitor();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void bookME(DQMStore::IBooker &, METplusTrackME& me, const std::string& histname, const std::string& histtitle, int nbins, double xmin, double xmax);
  void bookME(DQMStore::IBooker &, METplusTrackME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX);
  void bookME(DQMStore::IBooker &, METplusTrackME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, METplusTrackME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax, bool bookNumerator);
  void bookME(DQMStore::IBooker &, METplusTrackME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY);
  void setMETitle(METplusTrackME& me, std::string titleX, std::string titleY, bool bookNumerator);

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:

  static MEbinning getHistoPSet   (edm::ParameterSet pset);
  static MEbinning getHistoLSPSet (edm::ParameterSet pset);

  bool getHLTObj(const edm::Handle<trigger::TriggerEvent> &trigSummary, const edm::InputTag filterTag, trigger::TriggerObject &obj) const;

  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::CaloMETCollection> metToken_;
  edm::EDGetTokenT<reco::CaloMETCollection> hltMetToken_;
  edm::EDGetTokenT<reco::CaloMETCollection> hltMetCleanToken_;
  edm::EDGetTokenT<reco::MuonCollection>    muonToken_;
  edm::EDGetTokenT<reco::PFJetCollection>   jetToken_;
  edm::EDGetTokenT<reco::VertexCollection>  vtxToken_;
  edm::EDGetTokenT<trigger::TriggerEvent>   theTrigSummary_;

  edm::InputTag trackLegFilterTag_;

  std::vector<double> met_variable_binning_;
  std::vector<double> muonPt_variable_binning_;

  MEbinning met_binning_;
  MEbinning ls_binning_;
  MEbinning pt_binning_;
  MEbinning eta_binning_;
  MEbinning phi_binning_;

  METplusTrackME metME_variableBinning_;
  METplusTrackME metVsLS_;
  METplusTrackME metPhiME_;
  METplusTrackME deltaphimetj1ME_;
  METplusTrackME metVsHltMet_;
  METplusTrackME metVsHltMetClean_;

  METplusTrackME muonPtME_variableBinning_;
  METplusTrackME muonPtVsLS_;
  METplusTrackME muonEtaME_;
  METplusTrackME deltaphimetmuonME_;
  METplusTrackME muonEtaVsPhi_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::CaloMET, true>    metSelection_;
  StringCutObjectSelector<reco::CaloMET, true>    hltMetSelection_;
  StringCutObjectSelector<reco::CaloMET, true>    hltMetCleanSelection_;

  StringCutObjectSelector<reco::Muon, true>   muonSelection_;
  StringCutObjectSelector<reco::PFJet, true>  jetSelection_;
  StringCutObjectSelector<reco::Vertex, true> vtxSelection_;
  unsigned nmuons_;
  unsigned njets_;

  double leadJetEtaCut_;
  double hltMetCut_;
  double hltMetCleanCut_;

};


#endif // DQMOFFLINE_TRIGGER_METPLUSTRACKMONITOR_H

