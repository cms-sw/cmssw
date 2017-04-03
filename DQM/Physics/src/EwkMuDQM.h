#ifndef EwkMuDQM_H
#define EwkMuDQM_H

/** \class EwkMuDQM
 *
 *  DQM offline for EWKMu
 *
 */



#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"  //This is  added
#include "DQMServices/Core/interface/MonitorElement.h" //This is  added


#include "DataFormats/TrackReco/interface/Track.h" //This is  added
#include "DataFormats/BeamSpot/interface/BeamSpot.h" //This is  added
#include "DataFormats/VertexReco/interface/Vertex.h" //This is  added



#include "DataFormats/Candidate/interface/Candidate.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/Muon.h"



#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
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

class EwkMuDQM : public DQMEDAnalyzer {
 public:
  EwkMuDQM(const edm::ParameterSet&);
   ~EwkMuDQM();
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
  edm::EDGetTokenT<edm::View<reco::Muon> > muonTag_;
  edm::EDGetTokenT<edm::View<reco::MET> > metToken_;
  edm::EDGetTokenT<edm::View<reco::Jet> > jetToken_;
  edm::EDGetTokenT<edm::View<reco::Photon> > phoTag_;
  edm::EDGetTokenT<edm::View<reco::Vertex> > vertexTag_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotTag_;
  std::vector<std::string> trigPathNames_;
  int FindMassBin (double MassGrid[], double Mass, const int size);
  int FindRapBin (double RapGrid[], double Rap, const int size);

 


  bool isAlsoTrackerMuon_;
  double dxyCut_;
  double normalizedChi2Cut_;
  int trackerHitsCut_;
  int pixelHitsCut_;
  int muonHitsCut_;
  int nMatchesCut_;

  bool isRelativeIso_;
  bool isCombinedIso_;
  double isoCut03_;

  double acopCut_;
  double metMin_;
  double metMax_;
  double mtMin_;
  double mtMax_;

  double ptCut_;
  double etaCut_;

  double ptThrForZ1_;
  double ptThrForZ2_;

  double dimuonMassMin_;
  double dimuonMassMax_;

  double eJetMin_;
  int nJetMax_;

  double ptThrForPhoton_;
  int nPhoMax_;

  unsigned int nall;
  unsigned int nrec;
  unsigned int niso;
  unsigned int nhlt;
  unsigned int nmet;
  unsigned int nsel;
  unsigned int nz;

  MonitorElement* pt_before_;
  MonitorElement* pt_after_;
  MonitorElement* eta_before_;
  MonitorElement* eta_after_;
  MonitorElement* dxy_before_;
  MonitorElement* dxy_after_;
  MonitorElement* goodewkmuon_before_;
  MonitorElement* goodewkmuon_after_;
  MonitorElement* iso_before_;
  MonitorElement* iso_after_;
  MonitorElement* trig_before_;
  MonitorElement* trig_after_;
  MonitorElement* mt_before_;
  MonitorElement* mt_after_;
  MonitorElement* met_before_;
  MonitorElement* met_after_;
  MonitorElement* acop_before_;
  MonitorElement* acop_after_;

  MonitorElement* njets_before_;
  MonitorElement* njets_after_;
  MonitorElement* njets_afterZ_;
  MonitorElement* leadingjet_pt_before_;
  MonitorElement* leadingjet_pt_after_;
  MonitorElement* leadingjet_pt_afterZ_;
  MonitorElement* leadingjet_eta_before_;
  MonitorElement* leadingjet_eta_after_;
  MonitorElement* leadingjet_eta_afterZ_;

  MonitorElement* ptDiffPM_before_;
  MonitorElement* ptDiffPM_afterZ_;

  MonitorElement* met_afterZ_;
  MonitorElement* pt1_afterZ_;
  MonitorElement* eta1_afterZ_;
  MonitorElement* dxy1_afterZ_;
  MonitorElement* goodewkmuon1_afterZ_;
  MonitorElement* iso1_afterZ_;
  MonitorElement* pt2_afterZ_;
  MonitorElement* eta2_afterZ_;
  MonitorElement* dxy2_afterZ_;
  MonitorElement* goodewkmuon2_afterZ_;
  MonitorElement* iso2_afterZ_;

  // filled if there is a Z-candidate
  MonitorElement* n_zselPt1thr_;  // number of muons in the event with pt>pt1thr
  MonitorElement* n_zselPt2thr_;  // number of muons in the event with pt>pt2thr

  MonitorElement* ztrig_afterZ_;
  MonitorElement* dimuonmass_before_;
  MonitorElement* dimuonmass_afterZ_;

  MonitorElement* npvs_before_;
  MonitorElement* npvs_after_;
  MonitorElement* npvs_afterZ_;

  MonitorElement* muoncharge_before_;
  MonitorElement* muoncharge_after_;
  MonitorElement* muoncharge_afterZ_;

  MonitorElement* nmuons_;
  MonitorElement* ngoodmuons_;

  MonitorElement* npfph_;
  MonitorElement* nph_;
  MonitorElement* pfphPt_;
  MonitorElement* phPt_;
  MonitorElement* pfphEta_;
  MonitorElement* phEta_;
  
  MonitorElement*  jet_HT_ ;
  MonitorElement*  jet_HT1_;
  MonitorElement*  jet_HT2_;
  MonitorElement*  jet_HT_after_;
  MonitorElement*  jet_HT_afterZ_; 
  MonitorElement*  Phistar_;
  MonitorElement*  Phistar_afterZ_;
  MonitorElement*  CosineThetaStar_;
  MonitorElement*  CosineThetaStar_afterZ_;


 MonitorElement*  leadingjet_phi_before_;
 MonitorElement*  deltaPhi_;
 MonitorElement*  deltaPhi_after_;
 MonitorElement*  deltaPhi_afterZ_;
 MonitorElement*  subleadingjet_pt_before_;
 MonitorElement*  subleadingjet_phi_before_;
 MonitorElement*  subleadingjet_pt_after_;
 MonitorElement*  subleadingjet_pt_afterZ_;
 MonitorElement*  subleadingjet_eta_before_;
 MonitorElement*  thirdleadingjet_phi_before_;
 MonitorElement*  InVaMassJJ_;


 MonitorElement*  CosineThetaStar_2D[3];
 MonitorElement*  CosineThetaStar_Y_2D[3];
 MonitorElement*  CosineThetaStar_afterZ_2D[3];
 MonitorElement*  CosineThetaStar_Y_afterZ_2D[3]; 
 MonitorElement*  subleadingjet_eta_after_;
 MonitorElement*  subleadingjet_eta_afterZ_;
 MonitorElement*  thirdleadingjet_pt_before_;
 MonitorElement*  thirdleadingjet_eta_before_;
 MonitorElement*  thirdleadingjet_pt_after_;
 MonitorElement*  thirdleadingjet_eta_after_;
 MonitorElement*  dimuonpt_afterZ_;
 

 MonitorElement*  thirdleadingjet_pt_afterZ_;
 MonitorElement*  thirdleadingjet_eta_afterZ_;
 MonitorElement*  InVaMassJJ_afterZ_;
 MonitorElement*  dimuonpt_before_;

  const int ZMassBins = 4;
  double ZMassGrid[4] = {60,80,100,120};

};

#endif

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
