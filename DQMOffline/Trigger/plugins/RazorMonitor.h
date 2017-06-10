#ifndef RAZORMONITOR_H
#define RAZORMONITOR_H

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

//Hemispheres                                                                                                                                                                  
#include "HLTrigger/JetMET/interface/HLTRHemisphere.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "TLorentzVector.h"

class GenericTriggerEventFlag;

struct MEbinning {
  int nbins;
  double xmin;
  double xmax;
};

struct RazorME {
  MonitorElement* numerator;
  MonitorElement* denominator;
};
//
// class declaration
//

class RazorMonitor : public DQMEDAnalyzer 
{
public:
  RazorMonitor( const edm::ParameterSet& );
  ~RazorMonitor();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  //static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  //static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);

  static double CalcMR(TLorentzVector ja,TLorentzVector jb);
  static double CalcR(double MR, TLorentzVector ja,TLorentzVector jb, edm::Handle<std::vector<reco::PFMET> > met, const std::vector<math::XYZTLorentzVector>& muons);

protected:

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void bookME(DQMStore::IBooker &, RazorME& me, const std::string& histname, const std::string& histtitle, int nbins, double xmin, double xmax);
  void bookME(DQMStore::IBooker &, RazorME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX);
  void bookME(DQMStore::IBooker &, RazorME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, RazorME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, RazorME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY);
  void setMETitle(RazorME& me, std::string titleX, std::string titleY);

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:
  static MEbinning getHistoLSPSet  (edm::ParameterSet pset);

  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::PFMETCollection>       metToken_;
  edm::EDGetTokenT<reco::PFJetCollection>       jetToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> eleToken_;
  edm::EDGetTokenT<reco::MuonCollection>        muoToken_;
  edm::EDGetTokenT<std::vector<math::XYZTLorentzVector> > theHemispheres_;

  MEbinning           mr_binning_;
  MEbinning           rsq_binning_;
  MEbinning           dphiR_binning_;
  MEbinning           ls_binning_;

  RazorME MR_ME_;
  RazorME MR_Jet120_ME_;
  RazorME MRVsLS_;
  RazorME Rsq_ME_;
  RazorME Rsq_Jet120_ME_;
  RazorME RsqVsLS_;
  RazorME dPhiR_ME_;
  RazorME dPhiRVsLS_;

  RazorME MRVsRsq_ME_;
  RazorME MRVsRsq_Jet120_ME_;

  GenericTriggerEventFlag* num_genTriggerEventFlag_;
  GenericTriggerEventFlag* den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::MET,true>         metSelection_;
  StringCutObjectSelector<reco::PFJet,true   >    jetSelection_;
  StringCutObjectSelector<reco::GsfElectron,true> eleSelection_;
  StringCutObjectSelector<reco::Muon,true>        muoSelection_;
  int njets_;
  int nelectrons_;
  int nmuons_;

};

#endif // RAZORMONITOR_H
