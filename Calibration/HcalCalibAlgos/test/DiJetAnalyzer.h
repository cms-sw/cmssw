#ifndef DiJetAnalyzer_h
#define DiJetAnalyzer_h

// system include files
#include <memory>
#include <string>
#include <vector>
#include <set>
#include <map>

#include "TTree.h"
#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TClonesArray.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Utilities/interface/EDMException.h"

// forward declarations
class TH1D;
class TH2D;
class TFile;
class TTree;

//
// class declarations
//

class JetCorretPair : protected std::pair<const reco::PFJet*, double> {
public:
  JetCorretPair() {
    first = 0;
    second = 1.0;
  }
  JetCorretPair(const reco::PFJet* j, double s) {
    first = j;
    second = s;
  }
  ~JetCorretPair() {}

  inline const reco::PFJet* jet(void) const { return first; }
  inline void jet(const reco::PFJet* j) {
    first = j;
    return;
  }
  inline double scale(void) const { return second; }
  inline void scale(double d) {
    second = d;
    return;
  }

private:
};

class DiJetAnalyzer : public edm::EDAnalyzer {
public:
  explicit DiJetAnalyzer(const edm::ParameterSet&);
  ~DiJetAnalyzer();

private:
  virtual void beginJob();  //(const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  // parameters
  bool debug_;                    // print debug statements
  std::string pfJetCollName_;     // label for the PF jet collection
  std::string pfJetCorrName_;     // label for the PF jet correction service
  std::string hbheRecHitName_;    // label for HBHERecHits collection
  std::string hfRecHitName_;      // label for HFRecHit collection
  std::string hoRecHitName_;      // label for HORecHit collection
  std::string pvCollName_;        // label for primary vertex collection
  std::string rootHistFilename_;  // name of the histogram file
  double maxDeltaEta_;            // maximum delta-|Eta| between Jets
  double minTagJetEta_;           // minimum |eta| of the tag jet
  double maxTagJetEta_;           // maximum |eta| of the tag jet
  double minSumJetEt_;            // minimum Sum of the tag and probe jet Et
  double minJetEt_;               // minimum Jet Et
  double maxThirdJetEt_;          // maximum 3rd jet Et

  edm::EDGetTokenT<reco::PFJetCollection> tok_PFJet_;
  edm::EDGetTokenT<edm::SortedCollection<HBHERecHit, edm::StrictWeakOrdering<HBHERecHit> > > tok_HBHE_;
  edm::EDGetTokenT<edm::SortedCollection<HFRecHit, edm::StrictWeakOrdering<HFRecHit> > > tok_HF_;
  edm::EDGetTokenT<edm::SortedCollection<HORecHit, edm::StrictWeakOrdering<HORecHit> > > tok_HO_;
  edm::EDGetTokenT<reco::VertexCollection> tok_Vertex_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;

  // root file/histograms
  TFile* rootfile_;

  TH1D* h_PassSelPF_;
  TTree* tree_;

  float tpfjet_pt_, tpfjet_p_, tpfjet_E_, tpfjet_eta_, tpfjet_phi_, tpfjet_EMfrac_, tpfjet_hadEcalEfrac_, tpfjet_scale_,
      tpfjet_area_;
  int tpfjet_jetID_;
  float tpfjet_EBE_, tpfjet_EEE_, tpfjet_HBE_, tpfjet_HEE_, tpfjet_HFE_;
  float tpfjet_unkown_E_, tpfjet_unkown_px_, tpfjet_unkown_py_, tpfjet_unkown_pz_, tpfjet_unkown_EcalE_;
  float tpfjet_electron_E_, tpfjet_electron_px_, tpfjet_electron_py_, tpfjet_electron_pz_, tpfjet_electron_EcalE_;
  float tpfjet_muon_E_, tpfjet_muon_px_, tpfjet_muon_py_, tpfjet_muon_pz_, tpfjet_muon_EcalE_;
  float tpfjet_photon_E_, tpfjet_photon_px_, tpfjet_photon_py_, tpfjet_photon_pz_, tpfjet_photon_EcalE_;
  int tpfjet_unkown_n_, tpfjet_electron_n_, tpfjet_muon_n_, tpfjet_photon_n_;
  int tpfjet_had_n_;
  std::vector<float> tpfjet_had_E_, tpfjet_had_px_, tpfjet_had_py_, tpfjet_had_pz_, tpfjet_had_EcalE_,
      tpfjet_had_rawHcalE_, tpfjet_had_emf_;
  std::vector<int> tpfjet_had_id_, tpfjet_had_candtrackind_, tpfjet_had_ntwrs_;
  int tpfjet_ntwrs_;
  std::vector<int> tpfjet_twr_ieta_, tpfjet_twr_iphi_, tpfjet_twr_depth_, tpfjet_twr_subdet_, tpfjet_twr_candtrackind_,
      tpfjet_twr_hadind_, tpfjet_twr_elmttype_, tpfjet_twr_clusterind_;
  std::vector<float> tpfjet_twr_hade_, tpfjet_twr_frac_, tpfjet_twr_dR_;
  int tpfjet_cluster_n_;
  std::vector<float> tpfjet_cluster_eta_, tpfjet_cluster_phi_, tpfjet_cluster_dR_;
  int tpfjet_ncandtracks_;
  std::vector<float> tpfjet_candtrack_px_, tpfjet_candtrack_py_, tpfjet_candtrack_pz_, tpfjet_candtrack_EcalE_;
  float ppfjet_pt_, ppfjet_p_, ppfjet_E_, ppfjet_eta_, ppfjet_phi_, ppfjet_EMfrac_, ppfjet_hadEcalEfrac_, ppfjet_scale_,
      ppfjet_area_;
  int ppfjet_jetID_;
  float ppfjet_EBE_, ppfjet_EEE_, ppfjet_HBE_, ppfjet_HEE_, ppfjet_HFE_;
  float ppfjet_unkown_E_, ppfjet_unkown_px_, ppfjet_unkown_py_, ppfjet_unkown_pz_, ppfjet_unkown_EcalE_;
  float ppfjet_electron_E_, ppfjet_electron_px_, ppfjet_electron_py_, ppfjet_electron_pz_, ppfjet_electron_EcalE_;
  float ppfjet_muon_E_, ppfjet_muon_px_, ppfjet_muon_py_, ppfjet_muon_pz_, ppfjet_muon_EcalE_;
  float ppfjet_photon_E_, ppfjet_photon_px_, ppfjet_photon_py_, ppfjet_photon_pz_, ppfjet_photon_EcalE_;
  int ppfjet_unkown_n_, ppfjet_electron_n_, ppfjet_muon_n_, ppfjet_photon_n_;
  int ppfjet_had_n_;
  std::vector<float> ppfjet_had_E_, ppfjet_had_px_, ppfjet_had_py_, ppfjet_had_pz_, ppfjet_had_EcalE_,
      ppfjet_had_rawHcalE_, ppfjet_had_emf_;
  std::vector<int> ppfjet_had_id_, ppfjet_had_candtrackind_, ppfjet_had_ntwrs_;
  int ppfjet_ntwrs_;
  std::vector<int> ppfjet_twr_ieta_, ppfjet_twr_iphi_, ppfjet_twr_depth_, ppfjet_twr_subdet_, ppfjet_twr_candtrackind_,
      ppfjet_twr_hadind_, ppfjet_twr_elmttype_, ppfjet_twr_clusterind_;
  std::vector<float> ppfjet_twr_hade_, ppfjet_twr_frac_, ppfjet_twr_dR_;
  int ppfjet_cluster_n_;
  std::vector<float> ppfjet_cluster_eta_, ppfjet_cluster_phi_, ppfjet_cluster_dR_;
  int ppfjet_ncandtracks_;
  std::vector<float> ppfjet_candtrack_px_, ppfjet_candtrack_py_, ppfjet_candtrack_pz_, ppfjet_candtrack_EcalE_;
  float pf_dijet_deta_, pf_dijet_dphi_, pf_dijet_balance_;
  float pf_thirdjet_px_, pf_thirdjet_py_, pf_realthirdjet_px_, pf_realthirdjet_py_, pf_realthirdjet_scale_;
  int pf_Run_, pf_Lumi_, pf_Event_;
  int pf_NPV_;

  // helper functions
  double deltaR(const reco::Jet* j1, const reco::Jet* j2);
  double deltaR(const double eta1, const double phi1, const double eta2, const double phi2);
  int getEtaPhi(const DetId id);
  int getEtaPhi(const HcalDetId id);

  struct JetCorretPairComp {
    inline bool operator()(const JetCorretPair& a, const JetCorretPair& b) const {
      return (a.jet()->pt() * a.scale()) > (b.jet()->pt() * b.scale());
    }
  };
};

#endif
