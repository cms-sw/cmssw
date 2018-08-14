#ifndef OBJMONITOR_H
#define OBJMONITOR_H

#include <string>
#include <vector>
#include <map>
#include "TLorentzVector.h"

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

#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "DQMOffline/Trigger/plugins/METDQM.h"
#include "DQMOffline/Trigger/plugins/JetDQM.h"
#include "DQMOffline/Trigger/plugins/HTDQM.h"
#include "DQMOffline/Trigger/plugins/HMesonGammaDQM.h"


class GenericTriggerEventFlag;

//
// class declaration
//

class ObjMonitor : public DQMEDAnalyzer 
{
public:
  ObjMonitor( const edm::ParameterSet& );
  ~ObjMonitor() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:

  bool looseJetId(const double & abseta,
		  const double & NHF,
		  const double & NEMF,
		  const double & CHF,
		  const double & CEMF,
		  const unsigned & NumNeutralParticles,
		  const unsigned  & CHM);
  
  bool tightJetId(const double & abseta,
		  const double & NHF,
		  const double & NEMF,
		  const double & CHF,
		  const double & CEMF,
		  const unsigned & NumNeutralParticles,
		  const unsigned  & CHM);
  
  std::string folderName_;
  std::string histoSuffix_;

  edm::EDGetTokenT<reco::PFMETCollection>       metToken_;
  edm::EDGetTokenT<reco::PFJetCollection>       jetToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> eleToken_;
  edm::EDGetTokenT<reco::MuonCollection>        muoToken_;
  edm::EDGetTokenT<reco::PhotonCollection>      phoToken_;
  edm::EDGetTokenT<reco::TrackCollection>       trkToken_;

  //objects to plot
  //add your own with corresponding switch
  bool do_met_;
  METDQM metDQM_;
  bool do_jet_;
  JetDQM jetDQM_;
  bool do_ht_;
  HTDQM htDQM_;
  bool do_hmg_;
  HMesonGammaDQM hmgDQM_;


  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::MET,true>         metSelection_;
  StringCutObjectSelector<reco::PFJet,true   >    jetSelection_;
  std::string jetId_;
  StringCutObjectSelector<reco::PFJet,true   >    htjetSelection_;
  StringCutObjectSelector<reco::GsfElectron,true> eleSelection_;
  StringCutObjectSelector<reco::Muon,true>        muoSelection_;
  StringCutObjectSelector<reco::Photon,true>      phoSelection_;
  StringCutObjectSelector<reco::Track,true>       trkSelection_;

  unsigned njets_;
  unsigned nelectrons_;
  unsigned nmuons_;
  unsigned nphotons_;
  unsigned nmesons_;

};

#endif // OBJMONITOR_H
