#ifndef BPHMONITOR_H
#define BPHMONITOR_H

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

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/deltaR.h"

class GenericTriggerEventFlag;

struct MEbinning {
  int nbins;
  double xmin;
  double xmax;
};

struct  METME{
  MonitorElement* numerator;
  MonitorElement* denominator;
};

//
// class declaration
//

class BPHMonitor : public DQMEDAnalyzer 
{
public:
  BPHMonitor( const edm::ParameterSet& );
  ~BPHMonitor();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void bookME(DQMStore::IBooker &, METME& me, std::string& histname, std::string& histtitle, int& nbins, double& xmin, double& xmax);
  void bookME(DQMStore::IBooker &, METME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX);
  void bookME(DQMStore::IBooker &, METME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax, double& ymin, double& ymax);
  void bookME(DQMStore::IBooker &, METME& me, std::string& histname, std::string& histtitle, int& nbinsX, double& xmin, double& xmax, int& nbinsY, double& ymin, double& ymax);
  void bookME(DQMStore::IBooker &, METME& me, std::string& histname, std::string& histtitle, std::vector<double> binningX, std::vector<double> binningY);
  void setMETitle(METME& me, std::string titleX, std::string titleY);

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:
  static MEbinning getHistoPSet    (edm::ParameterSet pset);
  static MEbinning getHistoLSPSet  (edm::ParameterSet pset);

  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::MuonCollection>        muoToken_;
  edm::EDGetTokenT<reco::BeamSpot>        bsToken_;
  edm::EDGetTokenT<reco::VertexCollection>        PVsToken_;

  std::vector<double> met_variable_binning_;
  MEbinning           met_binning_;
  MEbinning           phi_binning_;
  MEbinning           pt_binning_;
  MEbinning           eta_binning_;
  MEbinning           d0_binning_;
  MEbinning           z0_binning_;
  MEbinning           ls_binning_;

  METME metME_variableBinning_;

  METME muPhi_;
  METME muEta_;
  METME muPt_;
  METME mud0_;
  METME muz0_;

  METME JpsiPhi_;
  METME JpsiEta_;
  METME JpsiPt_;
  METME JpsiM_;


  GenericTriggerEventFlag* num_genTriggerEventFlag_;
  GenericTriggerEventFlag* den_genTriggerEventFlag_;

//  StringCutObjectSelector<reco::MET,true>         metSelection_;
//  StringCutObjectSelector<reco::PFJet,true   >    jetSelection_;
// StringCutObjectSelector<reco::GsfElectron,true> eleSelection_;
  StringCutObjectSelector<reco::Muon,true>        muoSelection_;
  StringCutObjectSelector<reco::Muon,true>        muoSelection_ref;
//  int njets_;
//  int nelectrons_;
  int nmuons_;

};

#endif // METMONITOR_H
