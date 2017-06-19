#ifndef DISPLACEDJETTRACKMONITOR_H
#define DISPLACEDJETTRACKMONITOR_H

#include <string>
#include <vector>
#include <map>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

//DataFormats
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

//Delta R Calculation
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"



class GenericTriggerEventFlag;

struct MEbinning {
  int nbins;
  double xmin;
  double xmax;
};

struct DJME {
  MonitorElement* numerator;
  MonitorElement* denominator;
};
//
// class declaration
//

class DisplacedJetTrackMonitor : public DQMEDAnalyzer 
{
public:
  DisplacedJetTrackMonitor( const edm::ParameterSet& );
  ~DisplacedJetTrackMonitor();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void bookME(DQMStore::IBooker &, DJME& me, const std::string& histname, const std::string& histtitle, int nbins, double xmin, double xmax);
  void bookME(DQMStore::IBooker &, DJME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX);
  void bookME(DQMStore::IBooker &, DJME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, DJME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, DJME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY);
  void setMETitle(DJME& me, std::string titleX, std::string titleY);

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:
  static MEbinning getHistoPSet    (edm::ParameterSet pset);
  static MEbinning getHistoLSPSet  (edm::ParameterSet pset);

  std::string folderName_;
  std::string histoSuffix_;


  edm::EDGetTokenT<reco::CaloJetCollection>     calojetToken_;
  edm::EDGetTokenT<reco::TrackCollection>       tracksToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> eleToken_;
  edm::EDGetTokenT<reco::MuonCollection>        muoToken_;

  MEbinning           ntrack_binning_;
  MEbinning           ls_binning_;

  DJME nprompttrksjet1ME_;
  DJME nprompttrksjet2ME_;
  DJME nprompttrksjet1VsLS_;
  DJME nprompttrksjet2VsLS_;
  DJME ndisplacedtrksjet1ME_;
  DJME ndisplacedtrksjet2ME_;
  DJME ndisplacedtrksjet1VsLS_;
  DJME ndisplacedtrksjet2VsLS_;
  
   


  GenericTriggerEventFlag* num_genTriggerEventFlag_;
  GenericTriggerEventFlag* den_genTriggerEventFlag_;


  StringCutObjectSelector<reco::CaloJet, true >   calojetSelection_;
  StringCutObjectSelector<reco::Track,  true >    trackSelection_;
  StringCutObjectSelector<reco::GsfElectron, true> eleSelection_;
  StringCutObjectSelector<reco::Muon, true>        muoSelection_;

  int ncalojets_;
  int nelectrons_;
  int nmuons_;

};

#endif // DISPLACEDJETMONITOR_H
