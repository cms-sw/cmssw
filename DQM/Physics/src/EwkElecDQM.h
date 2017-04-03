#ifndef EwkElecDQM_H
#define EwkElecDQM_H

/** \class EwkElecDQM
 *
 *  DQM offline for EWKMu
 *
 */
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

namespace reco {
class Muon;
class Jet;
class MET;
class Vertex;
class Photon;
class BeamSpot;
}

class DQMStore;
class MonitorElement;

class EwkElecDQM : public DQMEDAnalyzer {
 public:
  EwkElecDQM(const edm::ParameterSet&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  //Book histograms
  void bookHistograms(DQMStore::IBooker &,
    edm::Run const &, edm::EventSetup const &) override;
  virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void endRun(const edm::Run&, const edm::EventSetup&) override;

  void init_histograms();

 private:
  edm::InputTag metTag_;
  edm::InputTag jetTag_;
  edm::EDGetTokenT<edm::TriggerResults> trigTag_;
  edm::EDGetTokenT<edm::View<reco::GsfElectron> > elecTag_;
  edm::EDGetTokenT<edm::View<reco::MET> > metToken_;
  edm::EDGetTokenT<edm::View<reco::Jet> > jetToken_;
  edm::EDGetTokenT<edm::View<reco::Vertex> > vertexTag_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotTag_;
  bool metIncludesMuons_;
  int FindMassBin (double MassGrid[], double Mass, const int size);
  int FindRapBin (double RapGrid[], double Rap, const int size);
  double calcDeltaPhi(double phi1,double phi2);

  const std::vector<std::string> elecTrig_;
  double ptCut_;
  double etaCut_;

  double sieieCutBarrel_;
  double sieieCutEndcap_;
  double detainCutBarrel_;
  double detainCutEndcap_;

  double ecalIsoCutBarrel_;
  double ecalIsoCutEndcap_;
  double hcalIsoCutBarrel_;
  double hcalIsoCutEndcap_;
  double trkIsoCutBarrel_;
  double trkIsoCutEndcap_;
  double mtMin_;
  double mtMax_;
  double metMin_;
  double metMax_;
  double eJetMin_;
  int nJetMax_;

  // PU dependence
  unsigned int PUMax_, PUBinCount_;

 
  unsigned int nall;
  unsigned int nrec;
  unsigned int neid;
  unsigned int niso;
  unsigned int nsel;
  unsigned int nGoodElectrons;

    MonitorElement* pt_before_;
     MonitorElement* pt_after_;
     MonitorElement* eta_before_;
     MonitorElement* eta_after_;
     MonitorElement* sieiebarrel_before_;
     MonitorElement* sieiebarrel_after_;
     MonitorElement* sieieendcap_before_;
     MonitorElement* sieieendcap_after_;
     MonitorElement* detainbarrel_before_;
     MonitorElement* detainbarrel_after_;
     MonitorElement* detainendcap_before_;
     MonitorElement* detainendcap_after_;
     MonitorElement* ecalisobarrel_before_;
     MonitorElement* ecalisobarrel_after_;
     MonitorElement* ecalisoendcap_before_;
     MonitorElement* ecalisoendcap_after_;
     MonitorElement* hcalisobarrel_before_;
     MonitorElement* hcalisobarrel_after_;
     MonitorElement* hcalisoendcap_before_;
     MonitorElement* hcalisoendcap_after_;
     MonitorElement* trkisobarrel_before_;
     MonitorElement* trkisobarrel_after_;
     MonitorElement* trkisoendcap_before_;
     MonitorElement* trkisoendcap_after_;
     MonitorElement* trig_before_;
     MonitorElement* trig_after_;
     MonitorElement* Phistar_;
     MonitorElement* Phistar_after_;
     MonitorElement* CosineThetaStar_;
     MonitorElement* CosineThetaStar_afterZ_;

     MonitorElement* CosineThetaStar_2D[3];
     MonitorElement* CosineThetaStar_afterZ_2D[3];
     MonitorElement* CosineThetaStar_Y_2D[3];
     MonitorElement* CosineThetaStar_Y_afterZ_2D[3];

     MonitorElement* deltaPhi_;
     MonitorElement* deltaPhi_afterZ_;

     MonitorElement* InVaMassJJ_;	
     MonitorElement* InVaMassJJ_after_;
     MonitorElement* invmass_before_;
     MonitorElement* invpt_before_;
     MonitorElement* invmass_after_;
     MonitorElement* invpt_after_;
     MonitorElement* invmassPU_before_;
     MonitorElement* invmassPU_afterZ_;
     MonitorElement* npvs_before_;
     MonitorElement* npvs_afterZ_;
     MonitorElement* nelectrons_before_;
     MonitorElement* nelectrons_after_;
     MonitorElement* mt_before_;
     MonitorElement* mt_after_;
     MonitorElement* met_before_;
     MonitorElement* met_after_;
     MonitorElement* njets_before_;
     MonitorElement* njets_after_;
     MonitorElement* jet_et_before_;
     MonitorElement* jet_et_after_;
     MonitorElement* jet2_et_before_;
     MonitorElement* jet2_et_after_;
     MonitorElement* jet3_et_before_;
     MonitorElement* jet3_et_after_;
     MonitorElement* jet_eta_before_;
     MonitorElement* jet_eta_after_;
  
  const int ZMassBins = 4;
  double ZMassGrid[4] = {60,80,100,120};
};
#endif
