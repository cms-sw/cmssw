//////////////////////////////////////////////////////////////////////
//                                                                  //
//  Analyzer for making mini-ntuple for L1 track performance plots  //
//                                                                  //
//////////////////////////////////////////////////////////////////////

////////////////////
// FRAMEWORK HEADERS
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

///////////////////////
// DATA FORMATS HEADERS
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

////////////////////////////
// DETECTOR GEOMETRY HEADERS
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"

////////////////
// PHYSICS TOOLS
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//My additions
#include "DataFormats/L1TCorrelator/interface/TkJet.h"
#include "DataFormats/L1TCorrelator/interface/TkJetFwd.h"
#include "DataFormats/L1Trigger/interface/TkJetWord.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/L1TCorrelator/interface/TkEtMiss.h"
#include "DataFormats/L1TCorrelator/interface/TkEtMissFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkHTMiss.h"
#include "DataFormats/L1TCorrelator/interface/TkHTMissFwd.h"
#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "L1Trigger/L1TTrackMatch/interface/L1TkEtMissEmuAlgo.h"
#include "L1Trigger/L1TTrackMatch/interface/L1TkHTMissEmulatorProducer.h"

///////////////
// ROOT HEADERS
#include <TROOT.h>
#include <TCanvas.h>
#include <TTree.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>
#include <TLorentzVector.h>

//////////////
// STD HEADERS
#include <memory>
#include <string>
#include <iostream>

//////////////
// NAMESPACES
using namespace std;
using namespace edm;

//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class L1TrackObjectNtupleMaker : public edm::one::EDAnalyzer<edm::one::SharedResources> {
private:
  // ----------constants, enums and typedefs ---------
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1Track;
  typedef edm::Ptr<L1Track> L1TrackPtr;
  typedef std::vector<L1TrackPtr> L1TrackPtrCollection;
  typedef std::vector<L1Track> L1TrackCollection;
  typedef edm::Ref<L1TrackCollection> L1TrackRef;
  typedef edm::RefVector<L1TrackCollection> L1TrackRefCollection;

public:
  // Constructor/destructor
  explicit L1TrackObjectNtupleMaker(const edm::ParameterSet& iConfig);
  ~L1TrackObjectNtupleMaker() override;

  // Mandatory methods
  void beginJob() override;
  void endJob() override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  // Other member functions
  int getSelectedTrackIndex(const L1TrackRef& trackRef,
                            const edm::Handle<L1TrackRefCollection>& selectedTrackRefs) const;

private:
  //-----------------------------------------------------------------------------------------------
  // Containers of parameters passed by python configuration file
  edm::ParameterSet config;

  int MyProcess;       // 11/13/211 for single electrons/muons/pions, 6/15 for pions from ttbar/taus, 1 for inclusive
  bool DebugMode;      // lots of debug printout statements
  bool SaveAllTracks;  // store in ntuples not only truth-matched tracks but ALL tracks
  bool SaveStubs;      // option to save also stubs in the ntuples (makes them large...)
  string Displaced;    // "Prompt", "Displaced", "Both"
  int TP_minNStub;  // require TPs to have >= minNStub (defining efficiency denominator) (==0 means to only require >= 1 cluster)
  int TP_minNStubLayer;  // require TPs to have stubs in >= minNStubLayer layers/disks (defining efficiency denominator)
  double TP_minPt;       // save TPs with pt > minPt
  double TP_maxEta;      // save TPs with |eta| < maxEta
  double TP_maxZ0;       // save TPs with |z0| < maxZ0
  int L1Tk_minNStub;     // require L1 tracks to have >= minNStub (this is mostly for tracklet purposes)
  bool SaveTrackJets;
  bool SaveTrackSums;

  edm::InputTag L1TrackInputTag;                                              // L1 track collection
  edm::InputTag MCTruthTrackInputTag;                                         // MC truth collection
  edm::InputTag L1TrackGTTInputTag;                                           // L1 track collection
  edm::InputTag L1TrackSelectedInputTag;                                      // L1 track collection
  edm::InputTag L1TrackSelectedEmulationInputTag;                             // L1 track collection
  edm::InputTag L1TrackSelectedAssociatedInputTag;                            // L1 track collection
  edm::InputTag L1TrackSelectedAssociatedEmulationInputTag;                   // L1 track collection
  edm::InputTag L1TrackSelectedForJetsInputTag;                               // L1 track collection
  edm::InputTag L1TrackSelectedEmulationForJetsInputTag;                      // L1 track collection
  edm::InputTag L1TrackSelectedAssociatedForJetsInputTag;                     // L1 track collection
  edm::InputTag L1TrackSelectedAssociatedEmulationForJetsInputTag;            // L1 track collection
  edm::InputTag L1TrackSelectedForEtMissInputTag;                             // L1 track collection
  edm::InputTag L1TrackSelectedEmulationForEtMissInputTag;                    // L1 track collection
  edm::InputTag L1TrackSelectedAssociatedForEtMissInputTag;                   // L1 track collection
  edm::InputTag L1TrackSelectedAssociatedEmulationForEtMissInputTag;          // L1 track collection
  edm::InputTag L1TrackExtendedInputTag;                                      // L1 track collection
  edm::InputTag MCTruthTrackExtendedInputTag;                                 // MC truth collection
  edm::InputTag L1TrackExtendedGTTInputTag;                                   // L1 track collection
  edm::InputTag L1TrackExtendedSelectedInputTag;                              // L1 track collection
  edm::InputTag L1TrackExtendedSelectedEmulationInputTag;                     // L1 track collection
  edm::InputTag L1TrackExtendedSelectedAssociatedInputTag;                    // L1 track collection
  edm::InputTag L1TrackExtendedSelectedAssociatedEmulationInputTag;           // L1 track collection
  edm::InputTag L1TrackExtendedSelectedForJetsInputTag;                       // L1 track collection
  edm::InputTag L1TrackExtendedSelectedEmulationForJetsInputTag;              // L1 track collection
  edm::InputTag L1TrackExtendedSelectedAssociatedForJetsInputTag;             // L1 track collection
  edm::InputTag L1TrackExtendedSelectedAssociatedEmulationForJetsInputTag;    // L1 track collection
  edm::InputTag L1TrackExtendedSelectedForEtMissInputTag;                     // L1 track collection
  edm::InputTag L1TrackExtendedSelectedEmulationForEtMissInputTag;            // L1 track collection
  edm::InputTag L1TrackExtendedSelectedAssociatedForEtMissInputTag;           // L1 track collection
  edm::InputTag L1TrackExtendedSelectedAssociatedEmulationForEtMissInputTag;  // L1 track collection
  edm::InputTag MCTruthClusterInputTag;
  edm::InputTag L1StubInputTag;
  edm::InputTag MCTruthStubInputTag;
  edm::InputTag TrackingParticleInputTag;
  edm::InputTag TrackingVertexInputTag;
  edm::InputTag GenJetInputTag;
  edm::InputTag RecoVertexInputTag;
  edm::InputTag RecoVertexEmuInputTag;
  edm::InputTag GenParticleInputTag;

  edm::InputTag TrackFastJetsInputTag;
  edm::InputTag TrackJetsInputTag;
  edm::InputTag TrackJetsEmuInputTag;
  edm::InputTag TrackMETInputTag;
  edm::InputTag TrackMETEmuInputTag;
  edm::InputTag TrackMHTInputTag;
  edm::InputTag TrackMHTEmuInputTag;

  edm::InputTag TrackFastJetsExtendedInputTag;
  edm::InputTag TrackJetsExtendedInputTag;
  edm::InputTag TrackJetsExtendedEmuInputTag;
  edm::InputTag TrackMETExtendedInputTag;
  //edm::InputTag TrackMETEmuExtendedInputTag;
  edm::InputTag TrackMHTExtendedInputTag;
  edm::InputTag TrackMHTEmuExtendedInputTag;

  edm::EDGetTokenT<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>> ttClusterToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> ttStubToken_;
  edm::EDGetTokenT<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>> ttClusterMCTruthToken_;
  edm::EDGetTokenT<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> ttStubMCTruthToken_;

  edm::EDGetTokenT<L1TrackCollection> ttTrackToken_;
  edm::EDGetTokenT<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>> ttTrackMCTruthToken_;
  edm::EDGetTokenT<L1TrackCollection> ttTrackGTTToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackSelectedToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackSelectedEmulationToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackSelectedAssociatedToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackSelectedAssociatedEmulationToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackSelectedForJetsToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackSelectedEmulationForJetsToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackSelectedAssociatedForJetsToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackSelectedAssociatedEmulationForJetsToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackSelectedForEtMissToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackSelectedEmulationForEtMissToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackSelectedAssociatedForEtMissToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackSelectedAssociatedEmulationForEtMissToken_;
  edm::EDGetTokenT<L1TrackCollection> ttTrackExtendedToken_;
  edm::EDGetTokenT<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>> ttTrackMCTruthExtendedToken_;
  edm::EDGetTokenT<L1TrackCollection> ttTrackExtendedGTTToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackExtendedSelectedToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackExtendedSelectedEmulationToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackExtendedSelectedAssociatedToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackExtendedSelectedAssociatedEmulationToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackExtendedSelectedForJetsToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackExtendedSelectedEmulationForJetsToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackExtendedSelectedAssociatedForJetsToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackExtendedSelectedAssociatedEmulationForJetsToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackExtendedSelectedForEtMissToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackExtendedSelectedEmulationForEtMissToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackExtendedSelectedAssociatedForEtMissToken_;
  edm::EDGetTokenT<L1TrackRefCollection> ttTrackExtendedSelectedAssociatedEmulationForEtMissToken_;

  edm::EDGetTokenT<std::vector<TrackingParticle>> TrackingParticleToken_;
  edm::EDGetTokenT<std::vector<TrackingVertex>> TrackingVertexToken_;
  edm::EDGetTokenT<std::vector<reco::GenJet>> GenJetToken_;
  edm::EDGetTokenT<std::vector<reco::GenParticle>> GenParticleToken_;
  edm::EDGetTokenT<l1t::VertexCollection> L1VertexToken_;
  edm::EDGetTokenT<l1t::VertexWordCollection> L1VertexEmuToken_;

  edm::EDGetTokenT<std::vector<l1t::TkJet>> TrackFastJetsToken_;
  edm::EDGetTokenT<std::vector<l1t::TkJet>> TrackFastJetsExtendedToken_;
  edm::EDGetTokenT<std::vector<l1t::TkEtMiss>> TrackMETToken_;
  edm::EDGetTokenT<std::vector<l1t::TkEtMiss>> TrackMETExtendedToken_;
  edm::EDGetTokenT<std::vector<l1t::EtSum>> TrackMETEmuToken_;
  //edm::EDGetTokenT<std::vector<l1t::TkEtMiss>> TrackMETEmuExtendedToken_;
  edm::EDGetTokenT<l1t::TkHTMissCollection> TrackMHTToken_;
  edm::EDGetTokenT<l1t::TkHTMissCollection> TrackMHTExtendedToken_;
  edm::EDGetTokenT<std::vector<l1t::EtSum>> TrackMHTEmuToken_;
  edm::EDGetTokenT<std::vector<l1t::EtSum>> TrackMHTEmuExtendedToken_;
  edm::EDGetTokenT<l1t::TkJetCollection> TrackJetsToken_;
  edm::EDGetTokenT<l1t::TkJetCollection> TrackJetsExtendedToken_;
  edm::EDGetTokenT<l1t::TkJetWordCollection> TrackJetsEmuToken_;
  edm::EDGetTokenT<l1t::TkJetWordCollection> TrackJetsExtendedEmuToken_;

  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeomToken_;

  //-----------------------------------------------------------------------------------------------
  // tree & branches for mini-ntuple
  bool available_;  // ROOT file for histograms is open.
  TTree* eventTree;

  // primary vertex
  std::vector<float>* m_pv_L1reco;
  std::vector<float>* m_pv_L1reco_sum;
  std::vector<float>* m_pv_L1reco_emu;
  std::vector<float>* m_pv_L1reco_sum_emu;
  std::vector<float>* m_pv_MC;
  std::vector<int>* m_MC_lep;

  //gen particles
  std::vector<float>* m_gen_pt;
  std::vector<float>* m_gen_phi;
  std::vector<float>* m_gen_pdgid;
  std::vector<float>* m_gen_z0;

  // all L1 tracks (prompt)
  std::vector<float>* m_trk_pt;
  std::vector<float>* m_trk_eta;
  std::vector<float>* m_trk_phi;
  std::vector<float>* m_trk_phi_local;
  std::vector<float>* m_trk_d0;  // (filled if nFitPar==5, else 999)
  std::vector<float>* m_trk_z0;
  std::vector<float>* m_trk_chi2;
  std::vector<float>* m_trk_chi2dof;
  std::vector<float>* m_trk_chi2rphi;
  std::vector<float>* m_trk_chi2rz;
  std::vector<float>* m_trk_bendchi2;
  std::vector<float>* m_trk_MVA1;
  std::vector<int>* m_trk_nstub;
  std::vector<int>* m_trk_lhits;
  std::vector<int>* m_trk_dhits;
  std::vector<int>* m_trk_seed;
  std::vector<int>* m_trk_hitpattern;
  std::vector<unsigned int>* m_trk_phiSector;
  std::vector<int>* m_trk_genuine;
  std::vector<int>* m_trk_loose;
  std::vector<int>* m_trk_unknown;
  std::vector<int>* m_trk_combinatoric;
  std::vector<int>* m_trk_fake;  //0 fake, 1 track from primary interaction, 2 secondary track
  std::vector<int>* m_trk_matchtp_pdgid;
  std::vector<float>* m_trk_matchtp_pt;
  std::vector<float>* m_trk_matchtp_eta;
  std::vector<float>* m_trk_matchtp_phi;
  std::vector<float>* m_trk_matchtp_z0;
  std::vector<float>* m_trk_matchtp_dxy;
  std::vector<float>* m_trk_gtt_pt;
  std::vector<float>* m_trk_gtt_eta;
  std::vector<float>* m_trk_gtt_phi;
  std::vector<int>* m_trk_selected_index;
  std::vector<int>* m_trk_selected_emulation_index;
  std::vector<int>* m_trk_selected_associated_index;
  std::vector<int>* m_trk_selected_associated_emulation_index;
  std::vector<int>* m_trk_selected_forjets_index;
  std::vector<int>* m_trk_selected_emulation_forjets_index;
  std::vector<int>* m_trk_selected_associated_forjets_index;
  std::vector<int>* m_trk_selected_associated_emulation_forjets_index;
  std::vector<int>* m_trk_selected_foretmiss_index;
  std::vector<int>* m_trk_selected_emulation_foretmiss_index;
  std::vector<int>* m_trk_selected_associated_foretmiss_index;
  std::vector<int>* m_trk_selected_associated_emulation_foretmiss_index;

  // all L1 tracks (extended)
  std::vector<float>* m_trkExt_pt;
  std::vector<float>* m_trkExt_eta;
  std::vector<float>* m_trkExt_phi;
  std::vector<float>* m_trkExt_phi_local;
  std::vector<float>* m_trkExt_d0;  // (filled if nFitPar==5, else 999)
  std::vector<float>* m_trkExt_z0;
  std::vector<float>* m_trkExt_chi2;
  std::vector<float>* m_trkExt_chi2dof;
  std::vector<float>* m_trkExt_chi2rphi;
  std::vector<float>* m_trkExt_chi2rz;
  std::vector<float>* m_trkExt_bendchi2;
  std::vector<float>* m_trkExt_MVA;
  std::vector<int>* m_trkExt_nstub;
  std::vector<int>* m_trkExt_lhits;
  std::vector<int>* m_trkExt_dhits;
  std::vector<int>* m_trkExt_seed;
  std::vector<int>* m_trkExt_hitpattern;
  std::vector<unsigned int>* m_trkExt_phiSector;
  std::vector<int>* m_trkExt_genuine;
  std::vector<int>* m_trkExt_loose;
  std::vector<int>* m_trkExt_unknown;
  std::vector<int>* m_trkExt_combinatoric;
  std::vector<int>* m_trkExt_fake;  //0 fake, 1 track from primary interaction, 2 secondary track
  std::vector<int>* m_trkExt_matchtp_pdgid;
  std::vector<float>* m_trkExt_matchtp_pt;
  std::vector<float>* m_trkExt_matchtp_eta;
  std::vector<float>* m_trkExt_matchtp_phi;
  std::vector<float>* m_trkExt_matchtp_z0;
  std::vector<float>* m_trkExt_matchtp_dxy;
  std::vector<float>* m_trkExt_gtt_pt;
  std::vector<float>* m_trkExt_gtt_eta;
  std::vector<float>* m_trkExt_gtt_phi;
  std::vector<int>* m_trkExt_selected_index;
  std::vector<int>* m_trkExt_selected_emulation_index;
  std::vector<int>* m_trkExt_selected_associated_index;
  std::vector<int>* m_trkExt_selected_associated_emulation_index;
  std::vector<int>* m_trkExt_selected_forjets_index;
  std::vector<int>* m_trkExt_selected_emulation_forjets_index;
  std::vector<int>* m_trkExt_selected_associated_forjets_index;
  std::vector<int>* m_trkExt_selected_associated_emulation_forjets_index;
  std::vector<int>* m_trkExt_selected_foretmiss_index;
  std::vector<int>* m_trkExt_selected_emulation_foretmiss_index;
  std::vector<int>* m_trkExt_selected_associated_foretmiss_index;
  std::vector<int>* m_trkExt_selected_associated_emulation_foretmiss_index;

  // all tracking particles
  std::vector<float>* m_tp_pt;
  std::vector<float>* m_tp_eta;
  std::vector<float>* m_tp_phi;
  std::vector<float>* m_tp_dxy;
  std::vector<float>* m_tp_d0;
  std::vector<float>* m_tp_z0;
  std::vector<float>* m_tp_d0_prod;
  std::vector<float>* m_tp_z0_prod;
  std::vector<int>* m_tp_pdgid;
  std::vector<int>* m_tp_nmatch;
  std::vector<int>* m_tp_nstub;
  std::vector<int>* m_tp_eventid;
  std::vector<int>* m_tp_charge;

  // *L1 track* properties if m_tp_nmatch > 0 (prompt)
  std::vector<float>* m_matchtrk_pt;
  std::vector<float>* m_matchtrk_eta;
  std::vector<float>* m_matchtrk_phi;
  std::vector<float>* m_matchtrk_d0;  //this variable is only filled if nFitPar==5
  std::vector<float>* m_matchtrk_z0;
  std::vector<float>* m_matchtrk_chi2;
  std::vector<float>* m_matchtrk_chi2dof;
  std::vector<float>* m_matchtrk_chi2rphi;
  std::vector<float>* m_matchtrk_chi2rz;
  std::vector<float>* m_matchtrk_bendchi2;
  std::vector<float>* m_matchtrk_MVA1;
  std::vector<int>* m_matchtrk_nstub;
  std::vector<int>* m_matchtrk_lhits;
  std::vector<int>* m_matchtrk_dhits;
  std::vector<int>* m_matchtrk_seed;
  std::vector<int>* m_matchtrk_hitpattern;

  // *L1 track* properties if m_tp_nmatch > 0 (extended)
  std::vector<float>* m_matchtrkExt_pt;
  std::vector<float>* m_matchtrkExt_eta;
  std::vector<float>* m_matchtrkExt_phi;
  std::vector<float>* m_matchtrkExt_d0;  //this variable is only filled if nFitPar==5
  std::vector<float>* m_matchtrkExt_z0;
  std::vector<float>* m_matchtrkExt_chi2;
  std::vector<float>* m_matchtrkExt_chi2dof;
  std::vector<float>* m_matchtrkExt_chi2rphi;
  std::vector<float>* m_matchtrkExt_chi2rz;
  std::vector<float>* m_matchtrkExt_bendchi2;
  std::vector<float>* m_matchtrkExt_MVA;
  std::vector<int>* m_matchtrkExt_nstub;
  std::vector<int>* m_matchtrkExt_lhits;
  std::vector<int>* m_matchtrkExt_dhits;
  std::vector<int>* m_matchtrkExt_seed;
  std::vector<int>* m_matchtrkExt_hitpattern;

  // ALL stubs
  std::vector<float>* m_allstub_x;
  std::vector<float>* m_allstub_y;
  std::vector<float>* m_allstub_z;
  std::vector<int>* m_allstub_isBarrel;  // stub is in barrel (1) or in disk (0)
  std::vector<int>* m_allstub_layer;
  std::vector<int>* m_allstub_isPSmodule;
  std::vector<float>* m_allstub_trigDisplace;
  std::vector<float>* m_allstub_trigOffset;
  std::vector<float>* m_allstub_trigPos;
  std::vector<float>* m_allstub_trigBend;

  // stub associated with tracking particle ?
  std::vector<int>* m_allstub_matchTP_pdgid;  // -999 if not matched
  std::vector<float>* m_allstub_matchTP_pt;   // -999 if not matched
  std::vector<float>* m_allstub_matchTP_eta;  // -999 if not matched
  std::vector<float>* m_allstub_matchTP_phi;  // -999 if not matched
  std::vector<int>* m_allstub_genuine;

  //prompt
  float trueMET = 0;
  float trueTkMET = 0;
  float trkMET = 0;
  float trkMETPhi = 0;
  float trkMHT = 0;
  float trkHT = 0;
  float trkMHTEmu = 0;
  float trkMHTEmuPhi = 0;
  float trkHTEmu = 0;
  float trkMETEmu = 0;
  float trkMETEmuPhi = 0;

  //displaced
  float trkMETExt = 0;
  float trkMETPhiExt = 0;
  float trkMHTExt = 0;
  float trkHTExt = 0;
  float trkMHTEmuExt = 0;
  float trkMHTEmuPhiExt = 0;
  float trkHTEmuExt = 0;

  //fast track jet
  std::vector<float>* m_trkfastjet_vz;
  std::vector<float>* m_trkfastjet_p;
  std::vector<float>* m_trkfastjet_phi;
  std::vector<float>* m_trkfastjet_eta;
  std::vector<float>* m_trkfastjet_pt;
  std::vector<int>* m_trkfastjet_ntracks;
  std::vector<float>* m_trkfastjet_tp_sumpt;
  std::vector<float>* m_trkfastjet_truetp_sumpt;

  std::vector<float>* m_trkfastjetExt_vz;
  std::vector<float>* m_trkfastjetExt_p;
  std::vector<float>* m_trkfastjetExt_phi;
  std::vector<float>* m_trkfastjetExt_eta;
  std::vector<float>* m_trkfastjetExt_pt;
  std::vector<int>* m_trkfastjetExt_ntracks;
  std::vector<float>* m_trkfastjetExt_tp_sumpt;
  std::vector<float>* m_trkfastjetExt_truetp_sumpt;

  std::vector<float>* m_trkjet_vz;
  std::vector<float>* m_trkjet_p;
  std::vector<float>* m_trkjet_phi;
  std::vector<float>* m_trkjet_eta;
  std::vector<float>* m_trkjet_pt;
  std::vector<int>* m_trkjet_ntracks;
  std::vector<int>* m_trkjet_nDisplaced;
  std::vector<int>* m_trkjet_nTight;
  std::vector<int>* m_trkjet_nTightDisplaced;
  std::vector<int>* m_trkjet_ntdtrk;

  std::vector<float>* m_trkjetem_pt;
  std::vector<float>* m_trkjetem_phi;
  std::vector<float>* m_trkjetem_eta;
  std::vector<float>* m_trkjetem_z;
  std::vector<int>* m_trkjetem_ntracks;
  std::vector<int>* m_trkjetem_nxtracks;

  std::vector<float>* m_trkjetExt_vz;
  std::vector<float>* m_trkjetExt_p;
  std::vector<float>* m_trkjetExt_phi;
  std::vector<float>* m_trkjetExt_eta;
  std::vector<float>* m_trkjetExt_pt;
  std::vector<int>* m_trkjetExt_ntracks;
  std::vector<int>* m_trkjetExt_nDisplaced;
  std::vector<int>* m_trkjetExt_nTight;
  std::vector<int>* m_trkjetExt_nTightDisplaced;
  std::vector<int>* m_trkjetExt_ntdtrk;

  std::vector<float>* m_trkjetemExt_pt;
  std::vector<float>* m_trkjetemExt_phi;
  std::vector<float>* m_trkjetemExt_eta;
  std::vector<float>* m_trkjetemExt_z;
  std::vector<int>* m_trkjetemExt_ntracks;
  std::vector<int>* m_trkjetemExt_nxtracks;
};

//////////////////////////////////
//                              //
//     CLASS IMPLEMENTATION     //
//                              //
//////////////////////////////////

//////////////
// CONSTRUCTOR
L1TrackObjectNtupleMaker::L1TrackObjectNtupleMaker(edm::ParameterSet const& iConfig) : config(iConfig) {
  MyProcess = iConfig.getParameter<int>("MyProcess");
  DebugMode = iConfig.getParameter<bool>("DebugMode");
  SaveAllTracks = iConfig.getParameter<bool>("SaveAllTracks");
  SaveStubs = iConfig.getParameter<bool>("SaveStubs");
  Displaced = iConfig.getParameter<string>("Displaced");
  TP_minNStub = iConfig.getParameter<int>("TP_minNStub");
  TP_minNStubLayer = iConfig.getParameter<int>("TP_minNStubLayer");
  TP_minPt = iConfig.getParameter<double>("TP_minPt");
  TP_maxEta = iConfig.getParameter<double>("TP_maxEta");
  TP_maxZ0 = iConfig.getParameter<double>("TP_maxZ0");
  L1Tk_minNStub = iConfig.getParameter<int>("L1Tk_minNStub");

  SaveTrackJets = iConfig.getParameter<bool>("SaveTrackJets");
  SaveTrackSums = iConfig.getParameter<bool>("SaveTrackSums");

  L1StubInputTag = iConfig.getParameter<edm::InputTag>("L1StubInputTag");
  MCTruthClusterInputTag = iConfig.getParameter<edm::InputTag>("MCTruthClusterInputTag");
  MCTruthStubInputTag = iConfig.getParameter<edm::InputTag>("MCTruthStubInputTag");
  TrackingParticleInputTag = iConfig.getParameter<edm::InputTag>("TrackingParticleInputTag");
  TrackingVertexInputTag = iConfig.getParameter<edm::InputTag>("TrackingVertexInputTag");
  GenJetInputTag = iConfig.getParameter<edm::InputTag>("GenJetInputTag");
  RecoVertexInputTag = iConfig.getParameter<InputTag>("RecoVertexInputTag");
  RecoVertexEmuInputTag = iConfig.getParameter<InputTag>("RecoVertexEmuInputTag");
  GenParticleInputTag = iConfig.getParameter<InputTag>("GenParticleInputTag");

  if (Displaced == "Prompt" || Displaced == "Both") {
    L1TrackInputTag = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");
    MCTruthTrackInputTag = iConfig.getParameter<edm::InputTag>("MCTruthTrackInputTag");
    L1TrackGTTInputTag = iConfig.getParameter<edm::InputTag>("L1TrackGTTInputTag");
    L1TrackSelectedInputTag = iConfig.getParameter<edm::InputTag>("L1TrackSelectedInputTag");
    L1TrackSelectedEmulationInputTag = iConfig.getParameter<edm::InputTag>("L1TrackSelectedEmulationInputTag");
    L1TrackSelectedAssociatedInputTag = iConfig.getParameter<edm::InputTag>("L1TrackSelectedAssociatedInputTag");
    L1TrackSelectedAssociatedEmulationInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackSelectedAssociatedEmulationInputTag");
    L1TrackSelectedForJetsInputTag = iConfig.getParameter<edm::InputTag>("L1TrackSelectedForJetsInputTag");
    L1TrackSelectedEmulationForJetsInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackSelectedEmulationForJetsInputTag");
    L1TrackSelectedAssociatedForJetsInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackSelectedAssociatedForJetsInputTag");
    L1TrackSelectedAssociatedEmulationForJetsInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackSelectedAssociatedEmulationForJetsInputTag");
    L1TrackSelectedForEtMissInputTag = iConfig.getParameter<edm::InputTag>("L1TrackSelectedForEtMissInputTag");
    L1TrackSelectedEmulationForEtMissInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackSelectedEmulationForEtMissInputTag");
    L1TrackSelectedAssociatedForEtMissInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackSelectedAssociatedForEtMissInputTag");
    L1TrackSelectedAssociatedEmulationForEtMissInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackSelectedAssociatedEmulationForEtMissInputTag");
    TrackFastJetsInputTag = iConfig.getParameter<InputTag>("TrackFastJetsInputTag");
    TrackJetsInputTag = iConfig.getParameter<InputTag>("TrackJetsInputTag");
    TrackJetsEmuInputTag = iConfig.getParameter<InputTag>("TrackJetsEmuInputTag");
    TrackMETInputTag = iConfig.getParameter<InputTag>("TrackMETInputTag");
    TrackMETEmuInputTag = iConfig.getParameter<InputTag>("TrackMETEmuInputTag");
    TrackMHTInputTag = iConfig.getParameter<InputTag>("TrackMHTInputTag");
    TrackMHTEmuInputTag = iConfig.getParameter<InputTag>("TrackMHTEmuInputTag");

    ttTrackToken_ = consumes<L1TrackCollection>(L1TrackInputTag);
    ttTrackMCTruthToken_ = consumes<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>>(MCTruthTrackInputTag);
    ttTrackGTTToken_ = consumes<L1TrackCollection>(L1TrackGTTInputTag);
    ttTrackSelectedToken_ = consumes<L1TrackRefCollection>(L1TrackSelectedInputTag);
    ttTrackSelectedEmulationToken_ = consumes<L1TrackRefCollection>(L1TrackSelectedEmulationInputTag);
    ttTrackSelectedAssociatedToken_ = consumes<L1TrackRefCollection>(L1TrackSelectedAssociatedInputTag);
    ttTrackSelectedAssociatedEmulationToken_ =
        consumes<L1TrackRefCollection>(L1TrackSelectedAssociatedEmulationInputTag);
    ttTrackSelectedForJetsToken_ = consumes<L1TrackRefCollection>(L1TrackSelectedForJetsInputTag);
    ttTrackSelectedEmulationForJetsToken_ = consumes<L1TrackRefCollection>(L1TrackSelectedEmulationForJetsInputTag);
    ttTrackSelectedAssociatedForJetsToken_ = consumes<L1TrackRefCollection>(L1TrackSelectedAssociatedForJetsInputTag);
    ttTrackSelectedAssociatedEmulationForJetsToken_ =
        consumes<L1TrackRefCollection>(L1TrackSelectedAssociatedEmulationForJetsInputTag);
    ttTrackSelectedForEtMissToken_ = consumes<L1TrackRefCollection>(L1TrackSelectedForEtMissInputTag);
    ttTrackSelectedEmulationForEtMissToken_ = consumes<L1TrackRefCollection>(L1TrackSelectedEmulationForEtMissInputTag);
    ttTrackSelectedAssociatedForEtMissToken_ =
        consumes<L1TrackRefCollection>(L1TrackSelectedAssociatedForEtMissInputTag);
    ttTrackSelectedAssociatedEmulationForEtMissToken_ =
        consumes<L1TrackRefCollection>(L1TrackSelectedAssociatedEmulationForEtMissInputTag);
    TrackFastJetsToken_ = consumes<std::vector<l1t::TkJet>>(TrackFastJetsInputTag);
    TrackJetsToken_ = consumes<l1t::TkJetCollection>(TrackJetsInputTag);
    TrackJetsEmuToken_ = consumes<l1t::TkJetWordCollection>(TrackJetsEmuInputTag);
    TrackMETToken_ = consumes<std::vector<l1t::TkEtMiss>>(TrackMETInputTag);
    TrackMETEmuToken_ = consumes<std::vector<l1t::EtSum>>(TrackMETEmuInputTag);
    TrackMHTToken_ = consumes<l1t::TkHTMissCollection>(TrackMHTInputTag);
    TrackMHTEmuToken_ = consumes<std::vector<l1t::EtSum>>(TrackMHTEmuInputTag);
  }

  if (Displaced == "Displaced" || Displaced == "Both") {
    L1TrackExtendedInputTag = iConfig.getParameter<edm::InputTag>("L1TrackExtendedInputTag");
    MCTruthTrackExtendedInputTag = iConfig.getParameter<edm::InputTag>("MCTruthTrackExtendedInputTag");
    L1TrackExtendedGTTInputTag = iConfig.getParameter<edm::InputTag>("L1TrackExtendedGTTInputTag");
    L1TrackExtendedSelectedInputTag = iConfig.getParameter<edm::InputTag>("L1TrackExtendedSelectedInputTag");
    L1TrackExtendedSelectedEmulationInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackExtendedSelectedEmulationInputTag");
    L1TrackExtendedSelectedAssociatedInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackExtendedSelectedAssociatedInputTag");
    L1TrackExtendedSelectedAssociatedEmulationInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackExtendedSelectedAssociatedEmulationInputTag");
    L1TrackExtendedSelectedForJetsInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackExtendedSelectedForJetsInputTag");
    L1TrackExtendedSelectedEmulationForJetsInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackExtendedSelectedEmulationForJetsInputTag");
    L1TrackExtendedSelectedAssociatedForJetsInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackExtendedSelectedAssociatedForJetsInputTag");
    L1TrackExtendedSelectedAssociatedEmulationForJetsInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackExtendedSelectedAssociatedEmulationForJetsInputTag");
    L1TrackExtendedSelectedForEtMissInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackExtendedSelectedForEtMissInputTag");
    L1TrackExtendedSelectedEmulationForEtMissInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackExtendedSelectedEmulationForEtMissInputTag");
    L1TrackExtendedSelectedAssociatedForEtMissInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackExtendedSelectedAssociatedForEtMissInputTag");
    L1TrackExtendedSelectedAssociatedEmulationForEtMissInputTag =
        iConfig.getParameter<edm::InputTag>("L1TrackExtendedSelectedAssociatedEmulationForEtMissInputTag");
    TrackFastJetsExtendedInputTag = iConfig.getParameter<InputTag>("TrackFastJetsExtendedInputTag");
    TrackJetsExtendedInputTag = iConfig.getParameter<InputTag>("TrackJetsExtendedInputTag");
    TrackJetsExtendedEmuInputTag = iConfig.getParameter<InputTag>("TrackJetsExtendedEmuInputTag");
    TrackMETExtendedInputTag = iConfig.getParameter<InputTag>("TrackMETExtendedInputTag");
    TrackMHTExtendedInputTag = iConfig.getParameter<InputTag>("TrackMHTExtendedInputTag");
    TrackMHTEmuInputTag = iConfig.getParameter<InputTag>("TrackMHTEmuInputTag");
    TrackMHTEmuExtendedInputTag = iConfig.getParameter<InputTag>("TrackMHTEmuExtendedInputTag");

    ttTrackExtendedToken_ = consumes<L1TrackCollection>(L1TrackExtendedInputTag);
    ttTrackMCTruthExtendedToken_ =
        consumes<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>>(MCTruthTrackExtendedInputTag);
    ttTrackExtendedGTTToken_ = consumes<L1TrackCollection>(L1TrackExtendedGTTInputTag);
    ttTrackExtendedSelectedToken_ = consumes<L1TrackRefCollection>(L1TrackExtendedSelectedInputTag);
    ttTrackExtendedSelectedEmulationToken_ = consumes<L1TrackRefCollection>(L1TrackExtendedSelectedEmulationInputTag);
    ttTrackExtendedSelectedAssociatedToken_ = consumes<L1TrackRefCollection>(L1TrackExtendedSelectedAssociatedInputTag);
    ttTrackExtendedSelectedAssociatedEmulationToken_ =
        consumes<L1TrackRefCollection>(L1TrackExtendedSelectedAssociatedEmulationInputTag);
    ttTrackExtendedSelectedForJetsToken_ = consumes<L1TrackRefCollection>(L1TrackExtendedSelectedForJetsInputTag);
    ttTrackExtendedSelectedEmulationForJetsToken_ =
        consumes<L1TrackRefCollection>(L1TrackExtendedSelectedEmulationForJetsInputTag);
    ttTrackExtendedSelectedAssociatedForJetsToken_ =
        consumes<L1TrackRefCollection>(L1TrackExtendedSelectedAssociatedForJetsInputTag);
    ttTrackExtendedSelectedAssociatedEmulationForJetsToken_ =
        consumes<L1TrackRefCollection>(L1TrackExtendedSelectedAssociatedEmulationForJetsInputTag);
    ttTrackExtendedSelectedForEtMissToken_ = consumes<L1TrackRefCollection>(L1TrackExtendedSelectedForEtMissInputTag);
    ttTrackExtendedSelectedEmulationForEtMissToken_ =
        consumes<L1TrackRefCollection>(L1TrackExtendedSelectedEmulationForEtMissInputTag);
    ttTrackExtendedSelectedAssociatedForEtMissToken_ =
        consumes<L1TrackRefCollection>(L1TrackExtendedSelectedAssociatedForEtMissInputTag);
    ttTrackExtendedSelectedAssociatedEmulationForEtMissToken_ =
        consumes<L1TrackRefCollection>(L1TrackExtendedSelectedAssociatedEmulationForEtMissInputTag);
    TrackFastJetsExtendedToken_ = consumes<std::vector<l1t::TkJet>>(TrackFastJetsExtendedInputTag);
    TrackJetsExtendedToken_ = consumes<l1t::TkJetCollection>(TrackJetsExtendedInputTag);
    TrackJetsExtendedEmuToken_ = consumes<l1t::TkJetWordCollection>(TrackJetsExtendedEmuInputTag);
    TrackMETExtendedToken_ = consumes<std::vector<l1t::TkEtMiss>>(TrackMETExtendedInputTag);
    TrackMHTExtendedToken_ = consumes<l1t::TkHTMissCollection>(TrackMHTExtendedInputTag);
    TrackMHTEmuToken_ = consumes<std::vector<l1t::EtSum>>(TrackMHTEmuInputTag);
    TrackMHTEmuExtendedToken_ = consumes<std::vector<l1t::EtSum>>(TrackMHTEmuExtendedInputTag);
  }

  ttStubToken_ = consumes<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>(L1StubInputTag);
  ttClusterMCTruthToken_ = consumes<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>>(MCTruthClusterInputTag);
  ttStubMCTruthToken_ = consumes<TTStubAssociationMap<Ref_Phase2TrackerDigi_>>(MCTruthStubInputTag);
  TrackingParticleToken_ = consumes<std::vector<TrackingParticle>>(TrackingParticleInputTag);
  TrackingVertexToken_ = consumes<std::vector<TrackingVertex>>(TrackingVertexInputTag);
  GenJetToken_ = consumes<std::vector<reco::GenJet>>(GenJetInputTag);
  GenParticleToken_ = consumes<std::vector<reco::GenParticle>>(GenParticleInputTag);
  L1VertexToken_ = consumes<l1t::VertexCollection>(RecoVertexInputTag);
  L1VertexEmuToken_ = consumes<l1t::VertexWordCollection>(RecoVertexEmuInputTag);
  tTopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>(edm::ESInputTag("", ""));
  tGeomToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>(edm::ESInputTag("", ""));

  usesResource(TFileService::kSharedResource);
}

/////////////
// DESTRUCTOR
L1TrackObjectNtupleMaker::~L1TrackObjectNtupleMaker() {}

//////////
// END JOB
void L1TrackObjectNtupleMaker::endJob() {
  // things to be done at the exit of the event Loop
  //  edm::LogVerbatim("Tracklet") << "L1TrackObjectNtupleMaker::endJob";
  // clean up raw pointers
  delete m_trk_pt;
  delete m_trk_eta;
  delete m_trk_phi;
  delete m_trk_phi_local;
  delete m_trk_z0;
  delete m_trk_d0;
  delete m_trk_chi2;
  delete m_trk_chi2dof;
  delete m_trk_chi2rphi;
  delete m_trk_chi2rz;
  delete m_trk_bendchi2;
  delete m_trk_MVA1;
  delete m_trk_nstub;
  delete m_trk_lhits;
  delete m_trk_dhits;
  delete m_trk_seed;
  delete m_trk_hitpattern;
  delete m_trk_phiSector;
  delete m_trk_genuine;
  delete m_trk_loose;
  delete m_trk_unknown;
  delete m_trk_combinatoric;
  delete m_trk_fake;
  delete m_trk_matchtp_pdgid;
  delete m_trk_matchtp_pt;
  delete m_trk_matchtp_eta;
  delete m_trk_matchtp_phi;
  delete m_trk_matchtp_z0;
  delete m_trk_matchtp_dxy;
  delete m_trk_gtt_pt;
  delete m_trk_gtt_eta;
  delete m_trk_gtt_phi;
  delete m_trk_selected_index;
  delete m_trk_selected_emulation_index;
  delete m_trk_selected_associated_index;
  delete m_trk_selected_associated_emulation_index;
  delete m_trk_selected_forjets_index;
  delete m_trk_selected_emulation_forjets_index;
  delete m_trk_selected_associated_forjets_index;
  delete m_trk_selected_associated_emulation_forjets_index;
  delete m_trk_selected_foretmiss_index;
  delete m_trk_selected_emulation_foretmiss_index;
  delete m_trk_selected_associated_foretmiss_index;
  delete m_trk_selected_associated_emulation_foretmiss_index;

  delete m_trkExt_pt;
  delete m_trkExt_eta;
  delete m_trkExt_phi;
  delete m_trkExt_phi_local;
  delete m_trkExt_z0;
  delete m_trkExt_d0;
  delete m_trkExt_chi2;
  delete m_trkExt_chi2dof;
  delete m_trkExt_chi2rphi;
  delete m_trkExt_chi2rz;
  delete m_trkExt_bendchi2;
  delete m_trkExt_MVA;
  delete m_trkExt_nstub;
  delete m_trkExt_lhits;
  delete m_trkExt_dhits;
  delete m_trkExt_seed;
  delete m_trkExt_hitpattern;
  delete m_trkExt_phiSector;
  delete m_trkExt_genuine;
  delete m_trkExt_loose;
  delete m_trkExt_unknown;
  delete m_trkExt_combinatoric;
  delete m_trkExt_fake;
  delete m_trkExt_matchtp_pdgid;
  delete m_trkExt_matchtp_pt;
  delete m_trkExt_matchtp_eta;
  delete m_trkExt_matchtp_phi;
  delete m_trkExt_matchtp_z0;
  delete m_trkExt_matchtp_dxy;
  delete m_trkExt_gtt_pt;
  delete m_trkExt_gtt_eta;
  delete m_trkExt_gtt_phi;
  delete m_trkExt_selected_index;
  delete m_trkExt_selected_emulation_index;
  delete m_trkExt_selected_associated_index;
  delete m_trkExt_selected_associated_emulation_index;
  delete m_trkExt_selected_forjets_index;
  delete m_trkExt_selected_emulation_forjets_index;
  delete m_trkExt_selected_associated_forjets_index;
  delete m_trkExt_selected_associated_emulation_forjets_index;
  delete m_trkExt_selected_foretmiss_index;
  delete m_trkExt_selected_emulation_foretmiss_index;
  delete m_trkExt_selected_associated_foretmiss_index;
  delete m_trkExt_selected_associated_emulation_foretmiss_index;

  delete m_tp_pt;
  delete m_tp_eta;
  delete m_tp_phi;
  delete m_tp_dxy;
  delete m_tp_d0;
  delete m_tp_z0;
  delete m_tp_d0_prod;
  delete m_tp_z0_prod;
  delete m_tp_pdgid;
  delete m_tp_nmatch;
  delete m_tp_nstub;
  delete m_tp_eventid;
  delete m_tp_charge;

  delete m_gen_pt;
  delete m_gen_phi;
  delete m_gen_pdgid;
  delete m_gen_z0;

  delete m_matchtrk_pt;
  delete m_matchtrk_eta;
  delete m_matchtrk_phi;
  delete m_matchtrk_z0;
  delete m_matchtrk_d0;
  delete m_matchtrk_chi2;
  delete m_matchtrk_chi2dof;
  delete m_matchtrk_chi2rphi;
  delete m_matchtrk_chi2rz;
  delete m_matchtrk_bendchi2;
  delete m_matchtrk_MVA1;
  delete m_matchtrk_nstub;
  delete m_matchtrk_dhits;
  delete m_matchtrk_lhits;
  delete m_matchtrk_seed;
  delete m_matchtrk_hitpattern;

  delete m_matchtrkExt_pt;
  delete m_matchtrkExt_eta;
  delete m_matchtrkExt_phi;
  delete m_matchtrkExt_z0;
  delete m_matchtrkExt_d0;
  delete m_matchtrkExt_chi2;
  delete m_matchtrkExt_chi2dof;
  delete m_matchtrkExt_chi2rphi;
  delete m_matchtrkExt_chi2rz;
  delete m_matchtrkExt_bendchi2;
  delete m_matchtrkExt_MVA;
  delete m_matchtrkExt_nstub;
  delete m_matchtrkExt_dhits;
  delete m_matchtrkExt_lhits;
  delete m_matchtrkExt_seed;
  delete m_matchtrkExt_hitpattern;

  delete m_allstub_x;
  delete m_allstub_y;
  delete m_allstub_z;
  delete m_allstub_isBarrel;
  delete m_allstub_layer;
  delete m_allstub_isPSmodule;
  delete m_allstub_trigDisplace;
  delete m_allstub_trigOffset;
  delete m_allstub_trigPos;
  delete m_allstub_trigBend;
  delete m_allstub_matchTP_pdgid;
  delete m_allstub_matchTP_pt;
  delete m_allstub_matchTP_eta;
  delete m_allstub_matchTP_phi;
  delete m_allstub_genuine;

  delete m_pv_L1reco;
  delete m_pv_L1reco_sum;
  delete m_pv_L1reco_emu;
  delete m_pv_L1reco_sum_emu;
  delete m_pv_MC;
  delete m_MC_lep;

  delete m_trkjet_eta;
  delete m_trkjet_vz;
  delete m_trkjet_phi;
  delete m_trkjet_p;
  delete m_trkjet_pt;
  delete m_trkjet_ntracks;
  delete m_trkjet_nDisplaced;
  delete m_trkjet_nTight;
  delete m_trkjet_nTightDisplaced;
  delete m_trkjet_ntdtrk;

  delete m_trkjetem_pt;
  delete m_trkjetem_phi;
  delete m_trkjetem_eta;
  delete m_trkjetem_z;
  delete m_trkjetem_ntracks;
  delete m_trkjetem_nxtracks;

  delete m_trkfastjet_eta;
  delete m_trkfastjet_vz;
  delete m_trkfastjet_phi;
  delete m_trkfastjet_p;
  delete m_trkfastjet_pt;
  delete m_trkfastjet_ntracks;
  delete m_trkfastjet_tp_sumpt;
  delete m_trkfastjet_truetp_sumpt;

  delete m_trkjetExt_eta;
  delete m_trkjetExt_vz;
  delete m_trkjetExt_phi;
  delete m_trkjetExt_p;
  delete m_trkjetExt_pt;
  delete m_trkjetExt_ntracks;
  delete m_trkjetExt_nDisplaced;
  delete m_trkjetExt_nTight;
  delete m_trkjetExt_nTightDisplaced;
  delete m_trkjetExt_ntdtrk;

  delete m_trkjetemExt_pt;
  delete m_trkjetemExt_phi;
  delete m_trkjetemExt_eta;
  delete m_trkjetemExt_z;
  delete m_trkjetemExt_ntracks;
  delete m_trkjetemExt_nxtracks;

  delete m_trkfastjetExt_eta;
  delete m_trkfastjetExt_vz;
  delete m_trkfastjetExt_phi;
  delete m_trkfastjetExt_p;
  delete m_trkfastjetExt_pt;
  delete m_trkfastjetExt_ntracks;
  delete m_trkfastjetExt_tp_sumpt;
  delete m_trkfastjetExt_truetp_sumpt;
}

////////////
// BEGIN JOB
void L1TrackObjectNtupleMaker::beginJob() {
  // things to be done before entering the event Loop
  //  edm::LogVerbatim("Tracklet") << "L1TrackObjectNtupleMaker::beginJob";

  //-----------------------------------------------------------------------------------------------
  // book histograms / make ntuple
  edm::Service<TFileService> fs;
  available_ = fs.isAvailable();
  if (not available_)
    return;  // No ROOT file open.

  // initilize
  m_trk_pt = new std::vector<float>;
  m_trk_eta = new std::vector<float>;
  m_trk_phi = new std::vector<float>;
  m_trk_phi_local = new std::vector<float>;
  m_trk_z0 = new std::vector<float>;
  m_trk_d0 = new std::vector<float>;
  m_trk_chi2 = new std::vector<float>;
  m_trk_chi2dof = new std::vector<float>;
  m_trk_chi2rphi = new std::vector<float>;
  m_trk_chi2rz = new std::vector<float>;
  m_trk_bendchi2 = new std::vector<float>;
  m_trk_MVA1 = new std::vector<float>;
  m_trk_nstub = new std::vector<int>;
  m_trk_lhits = new std::vector<int>;
  m_trk_dhits = new std::vector<int>;
  m_trk_seed = new std::vector<int>;
  m_trk_hitpattern = new std::vector<int>;
  m_trk_phiSector = new std::vector<unsigned int>;
  m_trk_genuine = new std::vector<int>;
  m_trk_loose = new std::vector<int>;
  m_trk_unknown = new std::vector<int>;
  m_trk_combinatoric = new std::vector<int>;
  m_trk_fake = new std::vector<int>;
  m_trk_matchtp_pdgid = new std::vector<int>;
  m_trk_matchtp_pt = new std::vector<float>;
  m_trk_matchtp_eta = new std::vector<float>;
  m_trk_matchtp_phi = new std::vector<float>;
  m_trk_matchtp_z0 = new std::vector<float>;
  m_trk_matchtp_dxy = new std::vector<float>;
  m_trk_gtt_pt = new std::vector<float>;
  m_trk_gtt_eta = new std::vector<float>;
  m_trk_gtt_phi = new std::vector<float>;
  m_trk_selected_index = new std::vector<int>;
  m_trk_selected_emulation_index = new std::vector<int>;
  m_trk_selected_associated_index = new std::vector<int>;
  m_trk_selected_associated_emulation_index = new std::vector<int>;
  m_trk_selected_forjets_index = new std::vector<int>;
  m_trk_selected_emulation_forjets_index = new std::vector<int>;
  m_trk_selected_associated_forjets_index = new std::vector<int>;
  m_trk_selected_associated_emulation_forjets_index = new std::vector<int>;
  m_trk_selected_foretmiss_index = new std::vector<int>;
  m_trk_selected_emulation_foretmiss_index = new std::vector<int>;
  m_trk_selected_associated_foretmiss_index = new std::vector<int>;
  m_trk_selected_associated_emulation_foretmiss_index = new std::vector<int>;

  m_trkExt_pt = new std::vector<float>;
  m_trkExt_eta = new std::vector<float>;
  m_trkExt_phi = new std::vector<float>;
  m_trkExt_phi_local = new std::vector<float>;
  m_trkExt_z0 = new std::vector<float>;
  m_trkExt_d0 = new std::vector<float>;
  m_trkExt_chi2 = new std::vector<float>;
  m_trkExt_chi2dof = new std::vector<float>;
  m_trkExt_chi2rphi = new std::vector<float>;
  m_trkExt_chi2rz = new std::vector<float>;
  m_trkExt_bendchi2 = new std::vector<float>;
  m_trkExt_MVA = new std::vector<float>;
  m_trkExt_nstub = new std::vector<int>;
  m_trkExt_lhits = new std::vector<int>;
  m_trkExt_dhits = new std::vector<int>;
  m_trkExt_seed = new std::vector<int>;
  m_trkExt_hitpattern = new std::vector<int>;
  m_trkExt_phiSector = new std::vector<unsigned int>;
  m_trkExt_genuine = new std::vector<int>;
  m_trkExt_loose = new std::vector<int>;
  m_trkExt_unknown = new std::vector<int>;
  m_trkExt_combinatoric = new std::vector<int>;
  m_trkExt_fake = new std::vector<int>;
  m_trkExt_matchtp_pdgid = new std::vector<int>;
  m_trkExt_matchtp_pt = new std::vector<float>;
  m_trkExt_matchtp_eta = new std::vector<float>;
  m_trkExt_matchtp_phi = new std::vector<float>;
  m_trkExt_matchtp_z0 = new std::vector<float>;
  m_trkExt_matchtp_dxy = new std::vector<float>;
  m_trkExt_gtt_pt = new std::vector<float>;
  m_trkExt_gtt_eta = new std::vector<float>;
  m_trkExt_gtt_phi = new std::vector<float>;
  m_trkExt_selected_index = new std::vector<int>;
  m_trkExt_selected_emulation_index = new std::vector<int>;
  m_trkExt_selected_associated_index = new std::vector<int>;
  m_trkExt_selected_associated_emulation_index = new std::vector<int>;
  m_trkExt_selected_forjets_index = new std::vector<int>;
  m_trkExt_selected_emulation_forjets_index = new std::vector<int>;
  m_trkExt_selected_associated_forjets_index = new std::vector<int>;
  m_trkExt_selected_associated_emulation_forjets_index = new std::vector<int>;
  m_trkExt_selected_foretmiss_index = new std::vector<int>;
  m_trkExt_selected_emulation_foretmiss_index = new std::vector<int>;
  m_trkExt_selected_associated_foretmiss_index = new std::vector<int>;
  m_trkExt_selected_associated_emulation_foretmiss_index = new std::vector<int>;

  m_tp_pt = new std::vector<float>;
  m_tp_eta = new std::vector<float>;
  m_tp_phi = new std::vector<float>;
  m_tp_dxy = new std::vector<float>;
  m_tp_d0 = new std::vector<float>;
  m_tp_z0 = new std::vector<float>;
  m_tp_d0_prod = new std::vector<float>;
  m_tp_z0_prod = new std::vector<float>;
  m_tp_pdgid = new std::vector<int>;
  m_tp_nmatch = new std::vector<int>;
  m_tp_nstub = new std::vector<int>;
  m_tp_eventid = new std::vector<int>;
  m_tp_charge = new std::vector<int>;

  m_gen_pt = new std::vector<float>;
  m_gen_phi = new std::vector<float>;
  m_gen_pdgid = new std::vector<float>;
  m_gen_z0 = new std::vector<float>;

  m_matchtrk_pt = new std::vector<float>;
  m_matchtrk_eta = new std::vector<float>;
  m_matchtrk_phi = new std::vector<float>;
  m_matchtrk_z0 = new std::vector<float>;
  m_matchtrk_d0 = new std::vector<float>;
  m_matchtrk_chi2 = new std::vector<float>;
  m_matchtrk_chi2dof = new std::vector<float>;
  m_matchtrk_chi2rphi = new std::vector<float>;
  m_matchtrk_chi2rz = new std::vector<float>;
  m_matchtrk_bendchi2 = new std::vector<float>;
  m_matchtrk_MVA1 = new std::vector<float>;
  m_matchtrk_nstub = new std::vector<int>;
  m_matchtrk_dhits = new std::vector<int>;
  m_matchtrk_lhits = new std::vector<int>;
  m_matchtrk_seed = new std::vector<int>;
  m_matchtrk_hitpattern = new std::vector<int>;

  m_matchtrkExt_pt = new std::vector<float>;
  m_matchtrkExt_eta = new std::vector<float>;
  m_matchtrkExt_phi = new std::vector<float>;
  m_matchtrkExt_z0 = new std::vector<float>;
  m_matchtrkExt_d0 = new std::vector<float>;
  m_matchtrkExt_chi2 = new std::vector<float>;
  m_matchtrkExt_chi2dof = new std::vector<float>;
  m_matchtrkExt_chi2rphi = new std::vector<float>;
  m_matchtrkExt_chi2rz = new std::vector<float>;
  m_matchtrkExt_bendchi2 = new std::vector<float>;
  m_matchtrkExt_MVA = new std::vector<float>;
  m_matchtrkExt_nstub = new std::vector<int>;
  m_matchtrkExt_dhits = new std::vector<int>;
  m_matchtrkExt_lhits = new std::vector<int>;
  m_matchtrkExt_seed = new std::vector<int>;
  m_matchtrkExt_hitpattern = new std::vector<int>;

  m_allstub_x = new std::vector<float>;
  m_allstub_y = new std::vector<float>;
  m_allstub_z = new std::vector<float>;
  m_allstub_isBarrel = new std::vector<int>;
  m_allstub_layer = new std::vector<int>;
  m_allstub_isPSmodule = new std::vector<int>;
  m_allstub_trigDisplace = new std::vector<float>;
  m_allstub_trigOffset = new std::vector<float>;
  m_allstub_trigPos = new std::vector<float>;
  m_allstub_trigBend = new std::vector<float>;
  m_allstub_matchTP_pdgid = new std::vector<int>;
  m_allstub_matchTP_pt = new std::vector<float>;
  m_allstub_matchTP_eta = new std::vector<float>;
  m_allstub_matchTP_phi = new std::vector<float>;
  m_allstub_genuine = new std::vector<int>;

  m_pv_L1reco = new std::vector<float>;
  m_pv_L1reco_sum = new std::vector<float>;
  m_pv_L1reco_emu = new std::vector<float>;
  m_pv_L1reco_sum_emu = new std::vector<float>;
  m_pv_MC = new std::vector<float>;
  m_MC_lep = new std::vector<int>;

  m_trkjet_eta = new std::vector<float>;
  m_trkjet_vz = new std::vector<float>;
  m_trkjet_phi = new std::vector<float>;
  m_trkjet_p = new std::vector<float>;
  m_trkjet_pt = new std::vector<float>;
  m_trkjet_ntracks = new std::vector<int>;
  m_trkjet_nDisplaced = new std::vector<int>;
  m_trkjet_nTight = new std::vector<int>;
  m_trkjet_nTightDisplaced = new std::vector<int>;
  m_trkjet_ntdtrk = new std::vector<int>;

  m_trkjetem_pt = new std::vector<float>;
  m_trkjetem_phi = new std::vector<float>;
  m_trkjetem_eta = new std::vector<float>;
  m_trkjetem_z = new std::vector<float>;
  m_trkjetem_ntracks = new std::vector<int>;
  m_trkjetem_nxtracks = new std::vector<int>;

  m_trkfastjet_eta = new std::vector<float>;
  m_trkfastjet_vz = new std::vector<float>;
  m_trkfastjet_phi = new std::vector<float>;
  m_trkfastjet_p = new std::vector<float>;
  m_trkfastjet_pt = new std::vector<float>;
  m_trkfastjet_ntracks = new std::vector<int>;
  m_trkfastjet_tp_sumpt = new std::vector<float>;
  m_trkfastjet_truetp_sumpt = new std::vector<float>;

  m_trkjetExt_eta = new std::vector<float>;
  m_trkjetExt_vz = new std::vector<float>;
  m_trkjetExt_phi = new std::vector<float>;
  m_trkjetExt_p = new std::vector<float>;
  m_trkjetExt_pt = new std::vector<float>;
  m_trkjetExt_ntracks = new std::vector<int>;
  m_trkjetExt_nDisplaced = new std::vector<int>;
  m_trkjetExt_nTight = new std::vector<int>;
  m_trkjetExt_nTightDisplaced = new std::vector<int>;
  m_trkjetExt_ntdtrk = new std::vector<int>;

  m_trkjetemExt_pt = new std::vector<float>;
  m_trkjetemExt_phi = new std::vector<float>;
  m_trkjetemExt_eta = new std::vector<float>;
  m_trkjetemExt_z = new std::vector<float>;
  m_trkjetemExt_ntracks = new std::vector<int>;
  m_trkjetemExt_nxtracks = new std::vector<int>;

  m_trkfastjetExt_eta = new std::vector<float>;
  m_trkfastjetExt_vz = new std::vector<float>;
  m_trkfastjetExt_phi = new std::vector<float>;
  m_trkfastjetExt_p = new std::vector<float>;
  m_trkfastjetExt_pt = new std::vector<float>;
  m_trkfastjetExt_ntracks = new std::vector<int>;
  m_trkfastjetExt_tp_sumpt = new std::vector<float>;
  m_trkfastjetExt_truetp_sumpt = new std::vector<float>;

  // ntuple
  eventTree = fs->make<TTree>("eventTree", "Event tree");
  if (SaveAllTracks && (Displaced == "Prompt" || Displaced == "Both")) {
    eventTree->Branch("trk_pt", &m_trk_pt);
    eventTree->Branch("trk_eta", &m_trk_eta);
    eventTree->Branch("trk_phi", &m_trk_phi);
    eventTree->Branch("trk_phi_local", &m_trk_phi_local);
    eventTree->Branch("trk_d0", &m_trk_d0);
    eventTree->Branch("trk_z0", &m_trk_z0);
    eventTree->Branch("trk_chi2", &m_trk_chi2);
    eventTree->Branch("trk_chi2dof", &m_trk_chi2dof);
    eventTree->Branch("trk_chi2rphi", &m_trk_chi2rphi);
    eventTree->Branch("trk_chi2rz", &m_trk_chi2rz);
    eventTree->Branch("trk_bendchi2", &m_trk_bendchi2);
    eventTree->Branch("trk_MVA1", &m_trk_MVA1);
    eventTree->Branch("trk_nstub", &m_trk_nstub);
    eventTree->Branch("trk_lhits", &m_trk_lhits);
    eventTree->Branch("trk_dhits", &m_trk_dhits);
    eventTree->Branch("trk_seed", &m_trk_seed);
    eventTree->Branch("trk_hitpattern", &m_trk_hitpattern);
    eventTree->Branch("trk_phiSector", &m_trk_phiSector);
    eventTree->Branch("trk_genuine", &m_trk_genuine);
    eventTree->Branch("trk_loose", &m_trk_loose);
    eventTree->Branch("trk_unknown", &m_trk_unknown);
    eventTree->Branch("trk_combinatoric", &m_trk_combinatoric);
    eventTree->Branch("trk_fake", &m_trk_fake);
    eventTree->Branch("trk_matchtp_pdgid", &m_trk_matchtp_pdgid);
    eventTree->Branch("trk_matchtp_pt", &m_trk_matchtp_pt);
    eventTree->Branch("trk_matchtp_eta", &m_trk_matchtp_eta);
    eventTree->Branch("trk_matchtp_phi", &m_trk_matchtp_phi);
    eventTree->Branch("trk_matchtp_z0", &m_trk_matchtp_z0);
    eventTree->Branch("trk_matchtp_dxy", &m_trk_matchtp_dxy);
    eventTree->Branch("trk_gtt_pt", &m_trk_gtt_pt);
    eventTree->Branch("trk_gtt_eta", &m_trk_gtt_eta);
    eventTree->Branch("trk_gtt_phi", &m_trk_gtt_phi);
    eventTree->Branch("trk_gtt_selected_index", &m_trk_selected_index);
    eventTree->Branch("trk_gtt_selected_emulation_index", &m_trk_selected_emulation_index);
    eventTree->Branch("trk_gtt_selected_associated_index", &m_trk_selected_associated_index);
    eventTree->Branch("trk_gtt_selected_associated_emulation_index", &m_trk_selected_associated_emulation_index);
    eventTree->Branch("trk_gtt_selected_forjets_index", &m_trk_selected_forjets_index);
    eventTree->Branch("trk_gtt_selected_emulation_forjets_index", &m_trk_selected_emulation_forjets_index);
    eventTree->Branch("trk_gtt_selected_associated_forjets_index", &m_trk_selected_associated_forjets_index);
    eventTree->Branch("trk_gtt_selected_associated_emulation_forjets_index",
                      &m_trk_selected_associated_emulation_forjets_index);
    eventTree->Branch("trk_gtt_selected_foretmiss_index", &m_trk_selected_foretmiss_index);
    eventTree->Branch("trk_gtt_selected_emulation_foretmiss_index", &m_trk_selected_emulation_foretmiss_index);
    eventTree->Branch("trk_gtt_selected_associated_foretmiss_index", &m_trk_selected_associated_foretmiss_index);
    eventTree->Branch("trk_gtt_selected_associated_emulation_foretmiss_index",
                      &m_trk_selected_associated_emulation_foretmiss_index);
  }

  if (SaveAllTracks && (Displaced == "Displaced" || Displaced == "Both")) {
    eventTree->Branch("trkExt_pt", &m_trkExt_pt);
    eventTree->Branch("trkExt_eta", &m_trkExt_eta);
    eventTree->Branch("trkExt_phi", &m_trkExt_phi);
    eventTree->Branch("trkExt_phi_local", &m_trkExt_phi_local);
    eventTree->Branch("trkExt_d0", &m_trkExt_d0);
    eventTree->Branch("trkExt_z0", &m_trkExt_z0);
    eventTree->Branch("trkExt_chi2", &m_trkExt_chi2);
    eventTree->Branch("trkExt_chi2dof", &m_trkExt_chi2dof);
    eventTree->Branch("trkExt_chi2rphi", &m_trkExt_chi2rphi);
    eventTree->Branch("trkExt_chi2rz", &m_trkExt_chi2rz);
    eventTree->Branch("trkExt_bendchi2", &m_trkExt_bendchi2);
    eventTree->Branch("trkExt_MVA", &m_trkExt_MVA);
    eventTree->Branch("trkExt_nstub", &m_trkExt_nstub);
    eventTree->Branch("trkExt_lhits", &m_trkExt_lhits);
    eventTree->Branch("trkExt_dhits", &m_trkExt_dhits);
    eventTree->Branch("trkExt_seed", &m_trkExt_seed);
    eventTree->Branch("trkExt_hitpattern", &m_trkExt_hitpattern);
    eventTree->Branch("trkExt_phiSector", &m_trkExt_phiSector);
    eventTree->Branch("trkExt_genuine", &m_trkExt_genuine);
    eventTree->Branch("trkExt_loose", &m_trkExt_loose);
    eventTree->Branch("trkExt_unknown", &m_trkExt_unknown);
    eventTree->Branch("trkExt_combinatoric", &m_trkExt_combinatoric);
    eventTree->Branch("trkExt_fake", &m_trkExt_fake);
    eventTree->Branch("trkExt_matchtp_pdgid", &m_trkExt_matchtp_pdgid);
    eventTree->Branch("trkExt_matchtp_pt", &m_trkExt_matchtp_pt);
    eventTree->Branch("trkExt_matchtp_eta", &m_trkExt_matchtp_eta);
    eventTree->Branch("trkExt_matchtp_phi", &m_trkExt_matchtp_phi);
    eventTree->Branch("trkExt_matchtp_z0", &m_trkExt_matchtp_z0);
    eventTree->Branch("trkExt_matchtp_dxy", &m_trkExt_matchtp_dxy);
    eventTree->Branch("trkExt_gtt_pt", &m_trkExt_gtt_pt);
    eventTree->Branch("trkExt_gtt_eta", &m_trkExt_gtt_eta);
    eventTree->Branch("trkExt_gtt_phi", &m_trkExt_gtt_phi);
    eventTree->Branch("trkExt_gtt_selected_index", &m_trkExt_selected_index);
    eventTree->Branch("trkExt_gtt_selected_emulation_index", &m_trkExt_selected_emulation_index);
    eventTree->Branch("trkExt_gtt_selected_associated_index", &m_trkExt_selected_associated_index);
    eventTree->Branch("trkExt_gtt_selected_associated_emulation_index", &m_trkExt_selected_associated_emulation_index);
    eventTree->Branch("trkExt_gtt_selected_forjets_index", &m_trkExt_selected_forjets_index);
    eventTree->Branch("trkExt_gtt_selected_emulation_forjets_index", &m_trkExt_selected_emulation_forjets_index);
    eventTree->Branch("trkExt_gtt_selected_associated_forjets_index", &m_trkExt_selected_associated_forjets_index);
    eventTree->Branch("trkExt_gtt_selected_associated_emulation_forjets_index",
                      &m_trkExt_selected_associated_emulation_forjets_index);
    eventTree->Branch("trkExt_gtt_selected_foretmiss_index", &m_trkExt_selected_foretmiss_index);
    eventTree->Branch("trkExt_gtt_selected_emulation_foretmiss_index", &m_trkExt_selected_emulation_foretmiss_index);
    eventTree->Branch("trkExt_gtt_selected_associated_foretmiss_index", &m_trkExt_selected_associated_foretmiss_index);
    eventTree->Branch("trkExt_gtt_selected_associated_emulation_foretmiss_index",
                      &m_trkExt_selected_associated_emulation_foretmiss_index);
  }
  eventTree->Branch("tp_pt", &m_tp_pt);
  eventTree->Branch("tp_eta", &m_tp_eta);
  eventTree->Branch("tp_phi", &m_tp_phi);
  eventTree->Branch("tp_dxy", &m_tp_dxy);
  eventTree->Branch("tp_d0", &m_tp_d0);
  eventTree->Branch("tp_z0", &m_tp_z0);
  eventTree->Branch("tp_d0_prod", &m_tp_d0_prod);
  eventTree->Branch("tp_z0_prod", &m_tp_z0_prod);
  eventTree->Branch("tp_pdgid", &m_tp_pdgid);
  eventTree->Branch("tp_nmatch", &m_tp_nmatch);
  eventTree->Branch("tp_nstub", &m_tp_nstub);
  eventTree->Branch("tp_eventid", &m_tp_eventid);
  eventTree->Branch("tp_charge", &m_tp_charge);

  if (Displaced == "Prompt" || Displaced == "Both") {
    eventTree->Branch("matchtrk_pt", &m_matchtrk_pt);
    eventTree->Branch("matchtrk_eta", &m_matchtrk_eta);
    eventTree->Branch("matchtrk_phi", &m_matchtrk_phi);
    eventTree->Branch("matchtrk_z0", &m_matchtrk_z0);
    eventTree->Branch("matchtrk_d0", &m_matchtrk_d0);
    eventTree->Branch("matchtrk_chi2", &m_matchtrk_chi2);
    eventTree->Branch("matchtrk_chi2dof", &m_matchtrk_chi2dof);
    eventTree->Branch("matchtrk_chi2rphi", &m_matchtrk_chi2rphi);
    eventTree->Branch("matchtrk_chi2rz", &m_matchtrk_chi2rz);
    eventTree->Branch("matchtrk_bendchi2", &m_matchtrk_bendchi2);
    eventTree->Branch("matchtrk_MVA1", &m_matchtrk_MVA1);
    eventTree->Branch("matchtrk_nstub", &m_matchtrk_nstub);
    eventTree->Branch("matchtrk_lhits", &m_matchtrk_lhits);
    eventTree->Branch("matchtrk_dhits", &m_matchtrk_dhits);
    eventTree->Branch("matchtrk_seed", &m_matchtrk_seed);
    eventTree->Branch("matchtrk_hitpattern", &m_matchtrk_hitpattern);
  }

  if (Displaced == "Displaced" || Displaced == "Both") {
    eventTree->Branch("matchtrkExt_pt", &m_matchtrkExt_pt);
    eventTree->Branch("matchtrkExt_eta", &m_matchtrkExt_eta);
    eventTree->Branch("matchtrkExt_phi", &m_matchtrkExt_phi);
    eventTree->Branch("matchtrkExt_z0", &m_matchtrkExt_z0);
    eventTree->Branch("matchtrkExt_d0", &m_matchtrkExt_d0);
    eventTree->Branch("matchtrkExt_chi2", &m_matchtrkExt_chi2);
    eventTree->Branch("matchtrkExt_chi2dof", &m_matchtrkExt_chi2dof);
    eventTree->Branch("matchtrkExt_chi2rphi", &m_matchtrkExt_chi2rphi);
    eventTree->Branch("matchtrkExt_chi2rz", &m_matchtrkExt_chi2rz);
    eventTree->Branch("matchtrkExt_bendchi2", &m_matchtrkExt_bendchi2);
    eventTree->Branch("matchtrkExt_MVA", &m_matchtrkExt_MVA);
    eventTree->Branch("matchtrkExt_nstub", &m_matchtrkExt_nstub);
    eventTree->Branch("matchtrkExt_lhits", &m_matchtrkExt_lhits);
    eventTree->Branch("matchtrkExt_dhits", &m_matchtrkExt_dhits);
    eventTree->Branch("matchtrkExt_seed", &m_matchtrkExt_seed);
    eventTree->Branch("matchtrkExt_hitpattern", &m_matchtrkExt_hitpattern);
  }

  if (SaveStubs) {
    eventTree->Branch("allstub_x", &m_allstub_x);
    eventTree->Branch("allstub_y", &m_allstub_y);
    eventTree->Branch("allstub_z", &m_allstub_z);
    eventTree->Branch("allstub_isBarrel", &m_allstub_isBarrel);
    eventTree->Branch("allstub_layer", &m_allstub_layer);
    eventTree->Branch("allstub_isPSmodule", &m_allstub_isPSmodule);
    eventTree->Branch("allstub_trigDisplace", &m_allstub_trigDisplace);
    eventTree->Branch("allstub_trigOffset", &m_allstub_trigOffset);
    eventTree->Branch("allstub_trigPos", &m_allstub_trigPos);
    eventTree->Branch("allstub_trigBend", &m_allstub_trigBend);
    eventTree->Branch("allstub_matchTP_pdgid", &m_allstub_matchTP_pdgid);
    eventTree->Branch("allstub_matchTP_pt", &m_allstub_matchTP_pt);
    eventTree->Branch("allstub_matchTP_eta", &m_allstub_matchTP_eta);
    eventTree->Branch("allstub_matchTP_phi", &m_allstub_matchTP_phi);
    eventTree->Branch("allstub_genuine", &m_allstub_genuine);
  }

  eventTree->Branch("pv_L1reco", &m_pv_L1reco);
  eventTree->Branch("pv_L1reco_sum", &m_pv_L1reco_sum);
  eventTree->Branch("pv_L1reco_emu", &m_pv_L1reco_emu);
  eventTree->Branch("pv_L1reco_sum_emu", &m_pv_L1reco_sum_emu);
  eventTree->Branch("MC_lep", &m_MC_lep);
  eventTree->Branch("pv_MC", &m_pv_MC);
  eventTree->Branch("gen_pt", &m_gen_pt);
  eventTree->Branch("gen_phi", &m_gen_phi);
  eventTree->Branch("gen_pdgid", &m_gen_pdgid);
  eventTree->Branch("gen_z0", &m_gen_z0);

  if (SaveTrackJets) {
    if (Displaced == "Prompt" || Displaced == "Both") {
      eventTree->Branch("trkfastjet_eta", &m_trkfastjet_eta);
      eventTree->Branch("trkfastjet_vz", &m_trkfastjet_vz);
      eventTree->Branch("trkfastjet_p", &m_trkfastjet_p);
      eventTree->Branch("trkfastjet_pt", &m_trkfastjet_pt);
      eventTree->Branch("trkfastjet_phi", &m_trkfastjet_phi);
      eventTree->Branch("trkfastjet_ntracks", &m_trkfastjet_ntracks);
      eventTree->Branch("trkfastjet_truetp_sumpt", m_trkfastjet_truetp_sumpt);

      eventTree->Branch("trkjet_eta", &m_trkjet_eta);
      eventTree->Branch("trkjet_vz", &m_trkjet_vz);
      eventTree->Branch("trkjet_p", &m_trkjet_p);
      eventTree->Branch("trkjet_pt", &m_trkjet_pt);
      eventTree->Branch("trkjet_phi", &m_trkjet_phi);
      eventTree->Branch("trkjet_ntracks", &m_trkjet_ntracks);
      eventTree->Branch("trkjet_nDisplaced", &m_trkjet_nDisplaced);
      eventTree->Branch("trkjet_nTight", &m_trkjet_nTight);
      eventTree->Branch("trkjet_nTightDisplaced", &m_trkjet_nTightDisplaced);

      eventTree->Branch("trkjetem_eta", &m_trkjetem_eta);
      eventTree->Branch("trkjetem_pt", &m_trkjetem_pt);
      eventTree->Branch("trkjetem_phi", &m_trkjetem_phi);
      eventTree->Branch("trkjetem_z", &m_trkjetem_z);
      eventTree->Branch("trkjetem_ntracks", &m_trkjetem_ntracks);
      eventTree->Branch("trkjetem_nxtracks", &m_trkjetem_nxtracks);
    }
    if (Displaced == "Displaced" || Displaced == "Both") {
      eventTree->Branch("trkfastjetExt_eta", &m_trkfastjetExt_eta);
      eventTree->Branch("trkfastjetExt_vz", &m_trkfastjetExt_vz);
      eventTree->Branch("trkfastjetExt_p", &m_trkfastjetExt_p);
      eventTree->Branch("trkfastjetExt_pt", &m_trkfastjetExt_pt);
      eventTree->Branch("trkfastjetExt_phi", &m_trkfastjetExt_phi);
      eventTree->Branch("trkfastjetExt_ntracks", &m_trkfastjetExt_ntracks);
      eventTree->Branch("trkfastjetExt_truetp_sumpt", m_trkfastjetExt_truetp_sumpt);

      eventTree->Branch("trkjetExt_eta", &m_trkjetExt_eta);
      eventTree->Branch("trkjetExt_vz", &m_trkjetExt_vz);
      eventTree->Branch("trkjetExt_p", &m_trkjetExt_p);
      eventTree->Branch("trkjetExt_pt", &m_trkjetExt_pt);
      eventTree->Branch("trkjetExt_phi", &m_trkjetExt_phi);
      eventTree->Branch("trkjetExt_ntracks", &m_trkjetExt_ntracks);
      eventTree->Branch("trkjetExt_nDisplaced", &m_trkjetExt_nDisplaced);
      eventTree->Branch("trkjetExt_nTight", &m_trkjetExt_nTight);
      eventTree->Branch("trkjetExt_nTightDisplaced", &m_trkjetExt_nTightDisplaced);

      eventTree->Branch("trkjetemExt_eta", &m_trkjetemExt_eta);
      eventTree->Branch("trkjetemExt_pt", &m_trkjetemExt_pt);
      eventTree->Branch("trkjetemExt_phi", &m_trkjetemExt_phi);
      eventTree->Branch("trkjetemExt_z", &m_trkjetemExt_z);
      eventTree->Branch("trkjetemExt_ntracks", &m_trkjetemExt_ntracks);
      eventTree->Branch("trkjetemExt_nxtracks", &m_trkjetemExt_nxtracks);
    }
  }

  if (SaveTrackSums) {
    eventTree->Branch("trueMET", &trueMET, "trueMET/F");
    eventTree->Branch("trueTkMET", &trueTkMET, "trueTkMET/F");

    if (Displaced == "Prompt" || Displaced == "Both") {
      eventTree->Branch("trkMET", &trkMET, "trkMET/F");
      eventTree->Branch("trkMETPhi", &trkMETPhi, "trkMETPhi/F");
      eventTree->Branch("trkMETEmu", &trkMETEmu, "trkMETEmu/F");
      eventTree->Branch("trkMETEmuPhi", &trkMETEmuPhi, "trkMETEmuPhi/F");
      eventTree->Branch("trkMHT", &trkMHT, "trkMHT/F");
      eventTree->Branch("trkHT", &trkHT, "trkHT/F");
      eventTree->Branch("trkMHTEmu", &trkMHTEmu, "trkMHTEmu/F");
      eventTree->Branch("trkMHTEmuPhi", &trkMHTEmuPhi, "trkMHTEmuPhi/F");
      eventTree->Branch("trkHTEmu", &trkHTEmu, "trkHTEmu/F");
    }
    if (Displaced == "Displaced" || Displaced == "Both") {
      eventTree->Branch("trkMETExt", &trkMETExt, "trkMETExt/F");
      eventTree->Branch("trkMETPhiExt", &trkMETPhiExt, "trkMETPhiExt/F");
      eventTree->Branch("trkMHTExt", &trkMHTExt, "trkMHTExt/F");
      eventTree->Branch("trkHTExt", &trkHTExt, "trkHTExt/F");
      eventTree->Branch("trkMHTEmuExt", &trkMHTEmuExt, "trkMHTEmuExt/F");
      eventTree->Branch("trkMHTEmuPhiExt", &trkMHTEmuPhiExt, "trkMHTEmuPhiExt/F");
      eventTree->Branch("trkHTEmuExt", &trkHTEmuExt, "trkHTEmuExt/F");
    }
  }
}

//////////
// ANALYZE
void L1TrackObjectNtupleMaker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (not available_)
    return;  // No ROOT file open.

  if (!(MyProcess == 13 || MyProcess == 11 || MyProcess == 211 || MyProcess == 6 || MyProcess == 15 ||
        MyProcess == 1)) {
    edm::LogVerbatim("Tracklet") << "The specified MyProcess is invalid! Exiting...";
    return;
  }

  // clear variables
  if (SaveAllTracks && (Displaced == "Prompt" || Displaced == "Both")) {
    m_trk_pt->clear();
    m_trk_eta->clear();
    m_trk_phi->clear();
    m_trk_phi_local->clear();
    m_trk_d0->clear();
    m_trk_z0->clear();
    m_trk_chi2->clear();
    m_trk_chi2dof->clear();
    m_trk_chi2rphi->clear();
    m_trk_chi2rz->clear();
    m_trk_bendchi2->clear();
    m_trk_MVA1->clear();
    m_trk_nstub->clear();
    m_trk_lhits->clear();
    m_trk_dhits->clear();
    m_trk_seed->clear();
    m_trk_hitpattern->clear();
    m_trk_phiSector->clear();
    m_trk_genuine->clear();
    m_trk_loose->clear();
    m_trk_unknown->clear();
    m_trk_combinatoric->clear();
    m_trk_fake->clear();
    m_trk_matchtp_pdgid->clear();
    m_trk_matchtp_pt->clear();
    m_trk_matchtp_eta->clear();
    m_trk_matchtp_phi->clear();
    m_trk_matchtp_z0->clear();
    m_trk_matchtp_dxy->clear();
    m_trk_gtt_pt->clear();
    m_trk_gtt_eta->clear();
    m_trk_gtt_phi->clear();
    m_trk_selected_index->clear();
    m_trk_selected_emulation_index->clear();
    m_trk_selected_associated_index->clear();
    m_trk_selected_associated_emulation_index->clear();
    m_trk_selected_forjets_index->clear();
    m_trk_selected_emulation_forjets_index->clear();
    m_trk_selected_associated_forjets_index->clear();
    m_trk_selected_associated_emulation_forjets_index->clear();
    m_trk_selected_foretmiss_index->clear();
    m_trk_selected_emulation_foretmiss_index->clear();
    m_trk_selected_associated_foretmiss_index->clear();
    m_trk_selected_associated_emulation_foretmiss_index->clear();
  }
  if (SaveAllTracks && (Displaced == "Displaced" || Displaced == "Both")) {
    m_trkExt_pt->clear();
    m_trkExt_eta->clear();
    m_trkExt_phi->clear();
    m_trkExt_phi_local->clear();
    m_trkExt_d0->clear();
    m_trkExt_z0->clear();
    m_trkExt_chi2->clear();
    m_trkExt_chi2dof->clear();
    m_trkExt_chi2rphi->clear();
    m_trkExt_chi2rz->clear();
    m_trkExt_bendchi2->clear();
    m_trkExt_MVA->clear();
    m_trkExt_nstub->clear();
    m_trkExt_lhits->clear();
    m_trkExt_dhits->clear();
    m_trkExt_seed->clear();
    m_trkExt_hitpattern->clear();
    m_trkExt_phiSector->clear();
    m_trkExt_genuine->clear();
    m_trkExt_loose->clear();
    m_trkExt_unknown->clear();
    m_trkExt_combinatoric->clear();
    m_trkExt_fake->clear();
    m_trkExt_matchtp_pdgid->clear();
    m_trkExt_matchtp_pt->clear();
    m_trkExt_matchtp_eta->clear();
    m_trkExt_matchtp_phi->clear();
    m_trkExt_matchtp_z0->clear();
    m_trkExt_matchtp_dxy->clear();
    m_trkExt_gtt_pt->clear();
    m_trkExt_gtt_eta->clear();
    m_trkExt_gtt_phi->clear();
    m_trkExt_selected_index->clear();
    m_trkExt_selected_emulation_index->clear();
    m_trkExt_selected_associated_index->clear();
    m_trkExt_selected_associated_emulation_index->clear();
    m_trkExt_selected_forjets_index->clear();
    m_trkExt_selected_emulation_forjets_index->clear();
    m_trkExt_selected_associated_forjets_index->clear();
    m_trkExt_selected_associated_emulation_forjets_index->clear();
    m_trkExt_selected_foretmiss_index->clear();
    m_trkExt_selected_emulation_foretmiss_index->clear();
    m_trkExt_selected_associated_foretmiss_index->clear();
    m_trkExt_selected_associated_emulation_foretmiss_index->clear();
  }
  m_tp_pt->clear();
  m_tp_eta->clear();
  m_tp_phi->clear();
  m_tp_dxy->clear();
  m_tp_d0->clear();
  m_tp_z0->clear();
  m_tp_d0_prod->clear();
  m_tp_z0_prod->clear();
  m_tp_pdgid->clear();
  m_tp_nmatch->clear();
  m_tp_nstub->clear();
  m_tp_eventid->clear();
  m_tp_charge->clear();

  m_gen_pt->clear();
  m_gen_phi->clear();
  m_gen_pdgid->clear();
  m_gen_z0->clear();

  if (Displaced == "Prompt" || Displaced == "Both") {
    m_matchtrk_pt->clear();
    m_matchtrk_eta->clear();
    m_matchtrk_phi->clear();
    m_matchtrk_z0->clear();
    m_matchtrk_d0->clear();
    m_matchtrk_chi2->clear();
    m_matchtrk_chi2dof->clear();
    m_matchtrk_chi2rphi->clear();
    m_matchtrk_chi2rz->clear();
    m_matchtrk_bendchi2->clear();
    m_matchtrk_MVA1->clear();
    m_matchtrk_nstub->clear();
    m_matchtrk_lhits->clear();
    m_matchtrk_dhits->clear();
    m_matchtrk_seed->clear();
    m_matchtrk_hitpattern->clear();
  }

  if (Displaced == "Displaced" || Displaced == "Both") {
    m_matchtrkExt_pt->clear();
    m_matchtrkExt_eta->clear();
    m_matchtrkExt_phi->clear();
    m_matchtrkExt_z0->clear();
    m_matchtrkExt_d0->clear();
    m_matchtrkExt_chi2->clear();
    m_matchtrkExt_chi2dof->clear();
    m_matchtrkExt_chi2rphi->clear();
    m_matchtrkExt_chi2rz->clear();
    m_matchtrkExt_bendchi2->clear();
    m_matchtrkExt_MVA->clear();
    m_matchtrkExt_nstub->clear();
    m_matchtrkExt_lhits->clear();
    m_matchtrkExt_dhits->clear();
    m_matchtrkExt_seed->clear();
    m_matchtrkExt_hitpattern->clear();
  }

  if (SaveStubs) {
    m_allstub_x->clear();
    m_allstub_y->clear();
    m_allstub_z->clear();
    m_allstub_isBarrel->clear();
    m_allstub_layer->clear();
    m_allstub_isPSmodule->clear();
    m_allstub_trigDisplace->clear();
    m_allstub_trigOffset->clear();
    m_allstub_trigPos->clear();
    m_allstub_trigBend->clear();
    m_allstub_matchTP_pdgid->clear();
    m_allstub_matchTP_pt->clear();
    m_allstub_matchTP_eta->clear();
    m_allstub_matchTP_phi->clear();
    m_allstub_genuine->clear();
  }

  if (SaveTrackJets) {
    if (Displaced == "Prompt" || Displaced == "Both") {
      m_trkjet_eta->clear();
      m_trkjet_pt->clear();
      m_trkjet_vz->clear();
      m_trkjet_phi->clear();
      m_trkjet_p->clear();
      m_trkjet_ntracks->clear();
      m_trkjet_nDisplaced->clear();
      m_trkjet_nTight->clear();
      m_trkjet_nTightDisplaced->clear();
      m_trkjet_ntdtrk->clear();
      m_trkfastjet_eta->clear();
      m_trkfastjet_pt->clear();
      m_trkfastjet_vz->clear();
      m_trkfastjet_phi->clear();
      m_trkfastjet_p->clear();
      m_trkfastjet_ntracks->clear();
      m_trkfastjet_truetp_sumpt->clear();
      m_trkfastjet_tp_sumpt->clear();
      m_trkjetem_eta->clear();
      m_trkjetem_pt->clear();
      m_trkjetem_phi->clear();
      m_trkjetem_z->clear();
      m_trkjetem_ntracks->clear();
      m_trkjetem_nxtracks->clear();
    }
    if (Displaced == "Displaced" || Displaced == "Both") {
      m_trkjetExt_eta->clear();
      m_trkjetExt_pt->clear();
      m_trkjetExt_vz->clear();
      m_trkjetExt_phi->clear();
      m_trkjetExt_p->clear();
      m_trkjetExt_ntracks->clear();
      m_trkjetExt_nDisplaced->clear();
      m_trkjetExt_nTight->clear();
      m_trkjetExt_nTightDisplaced->clear();
      m_trkjetExt_ntdtrk->clear();
      m_trkfastjetExt_eta->clear();
      m_trkfastjetExt_pt->clear();
      m_trkfastjetExt_vz->clear();
      m_trkfastjetExt_phi->clear();
      m_trkfastjetExt_p->clear();
      m_trkfastjetExt_ntracks->clear();
      m_trkfastjetExt_truetp_sumpt->clear();
      m_trkfastjetExt_tp_sumpt->clear();
      m_trkjetemExt_eta->clear();
      m_trkjetemExt_pt->clear();
      m_trkjetemExt_phi->clear();
      m_trkjetemExt_z->clear();
      m_trkjetemExt_ntracks->clear();
      m_trkjetemExt_nxtracks->clear();
    }

    m_pv_L1reco->clear();
    m_pv_L1reco_sum->clear();
    m_pv_L1reco_emu->clear();
    m_pv_L1reco_sum_emu->clear();
    m_pv_MC->clear();
    m_MC_lep->clear();
  }

  // -----------------------------------------------------------------------------------------------
  // retrieve various containers
  // -----------------------------------------------------------------------------------------------

  // L1 stubs
  edm::Handle<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> TTStubHandle;
  if (SaveStubs)
    iEvent.getByToken(ttStubToken_, TTStubHandle);

  // MC truth association maps
  edm::Handle<TTClusterAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTClusterHandle;
  iEvent.getByToken(ttClusterMCTruthToken_, MCTruthTTClusterHandle);
  edm::Handle<TTStubAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTStubHandle;
  iEvent.getByToken(ttStubMCTruthToken_, MCTruthTTStubHandle);

  // tracking particles
  edm::Handle<std::vector<TrackingParticle>> TrackingParticleHandle;
  edm::Handle<std::vector<TrackingVertex>> TrackingVertexHandle;
  iEvent.getByToken(TrackingParticleToken_, TrackingParticleHandle);
  iEvent.getByToken(TrackingVertexToken_, TrackingVertexHandle);

  // -----------------------------------------------------------------------------------------------
  // more for TTStubs
  const TrackerTopology& tTopo = iSetup.getData(tTopoToken_);
  const TrackerGeometry& tGeom = iSetup.getData(tGeomToken_);

  //Gen particles
  edm::Handle<std::vector<reco::GenParticle>> GenParticleHandle;
  iEvent.getByToken(GenParticleToken_, GenParticleHandle);

  //Vertex
  edm::Handle<l1t::VertexCollection> L1PrimaryVertexHandle;
  iEvent.getByToken(L1VertexToken_, L1PrimaryVertexHandle);
  std::vector<l1t::Vertex>::const_iterator vtxIter;

  edm::Handle<l1t::VertexWordCollection> L1PrimaryVertexEmuHandle;
  iEvent.getByToken(L1VertexEmuToken_, L1PrimaryVertexEmuHandle);
  std::vector<l1t::VertexWord>::const_iterator vtxEmuIter;

  // Track jets
  edm::Handle<std::vector<l1t::TkJet>> TrackFastJetsHandle;
  edm::Handle<std::vector<l1t::TkJet>> TrackFastJetsExtendedHandle;
  edm::Handle<l1t::TkJetCollection> TrackJetsHandle;
  edm::Handle<l1t::TkJetCollection> TrackJetsExtendedHandle;
  edm::Handle<l1t::TkJetWordCollection> TrackJetsEmuHandle;
  edm::Handle<l1t::TkJetWordCollection> TrackJetsExtendedEmuHandle;
  std::vector<l1t::TkJet>::const_iterator jetIter;
  std::vector<l1t::TkJetWord>::const_iterator jetemIter;

  // Track Sums
  edm::Handle<std::vector<l1t::TkEtMiss>> L1TkMETHandle;
  edm::Handle<std::vector<l1t::TkEtMiss>> L1TkMETExtendedHandle;
  edm::Handle<std::vector<l1t::EtSum>> L1TkMETEmuHandle;
  edm::Handle<std::vector<l1t::TkHTMiss>> L1TkMHTHandle;
  edm::Handle<std::vector<l1t::TkHTMiss>> L1TkMHTExtendedHandle;
  edm::Handle<std::vector<l1t::EtSum>> L1TkMHTEmuHandle;
  edm::Handle<std::vector<l1t::EtSum>> L1TkMHTEmuExtendedHandle;

  // L1 tracks
  edm::Handle<L1TrackCollection> TTTrackHandle;
  edm::Handle<L1TrackCollection> TTTrackExtendedHandle;
  edm::Handle<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTTrackHandle;
  edm::Handle<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTTrackExtendedHandle;
  edm::Handle<L1TrackCollection> TTTrackGTTHandle;
  edm::Handle<L1TrackCollection> TTTrackExtendedGTTHandle;
  edm::Handle<L1TrackRefCollection> TTTrackSelectedHandle;
  edm::Handle<L1TrackRefCollection> TTTrackSelectedEmulationHandle;
  edm::Handle<L1TrackRefCollection> TTTrackSelectedAssociatedHandle;
  edm::Handle<L1TrackRefCollection> TTTrackSelectedAssociatedEmulationHandle;
  edm::Handle<L1TrackRefCollection> TTTrackSelectedForJetsHandle;
  edm::Handle<L1TrackRefCollection> TTTrackSelectedEmulationForJetsHandle;
  edm::Handle<L1TrackRefCollection> TTTrackSelectedAssociatedForJetsHandle;
  edm::Handle<L1TrackRefCollection> TTTrackSelectedAssociatedEmulationForJetsHandle;
  edm::Handle<L1TrackRefCollection> TTTrackSelectedForEtMissHandle;
  edm::Handle<L1TrackRefCollection> TTTrackSelectedEmulationForEtMissHandle;
  edm::Handle<L1TrackRefCollection> TTTrackSelectedAssociatedForEtMissHandle;
  edm::Handle<L1TrackRefCollection> TTTrackSelectedAssociatedEmulationForEtMissHandle;
  edm::Handle<L1TrackRefCollection> TTTrackExtendedSelectedAssociatedHandle;
  edm::Handle<L1TrackRefCollection> TTTrackExtendedSelectedAssociatedEmulationHandle;
  edm::Handle<L1TrackRefCollection> TTTrackExtendedSelectedHandle;
  edm::Handle<L1TrackRefCollection> TTTrackExtendedSelectedEmulationHandle;
  edm::Handle<L1TrackRefCollection> TTTrackExtendedSelectedAssociatedForJetsHandle;
  edm::Handle<L1TrackRefCollection> TTTrackExtendedSelectedAssociatedEmulationForJetsHandle;
  edm::Handle<L1TrackRefCollection> TTTrackExtendedSelectedForJetsHandle;
  edm::Handle<L1TrackRefCollection> TTTrackExtendedSelectedEmulationForJetsHandle;
  edm::Handle<L1TrackRefCollection> TTTrackExtendedSelectedAssociatedForEtMissHandle;
  edm::Handle<L1TrackRefCollection> TTTrackExtendedSelectedAssociatedEmulationForEtMissHandle;
  edm::Handle<L1TrackRefCollection> TTTrackExtendedSelectedForEtMissHandle;
  edm::Handle<L1TrackRefCollection> TTTrackExtendedSelectedEmulationForEtMissHandle;
  L1TrackCollection::const_iterator iterL1Track;

  if (Displaced == "Prompt" || Displaced == "Both") {
    iEvent.getByToken(TrackFastJetsToken_, TrackFastJetsHandle);
    iEvent.getByToken(TrackJetsToken_, TrackJetsHandle);
    iEvent.getByToken(TrackJetsEmuToken_, TrackJetsEmuHandle);
    iEvent.getByToken(TrackMETToken_, L1TkMETHandle);
    iEvent.getByToken(TrackMETEmuToken_, L1TkMETEmuHandle);
    iEvent.getByToken(TrackMHTToken_, L1TkMHTHandle);
    iEvent.getByToken(TrackMHTEmuToken_, L1TkMHTEmuHandle);
    iEvent.getByToken(ttTrackToken_, TTTrackHandle);
    iEvent.getByToken(ttTrackMCTruthToken_, MCTruthTTTrackHandle);
    iEvent.getByToken(ttTrackGTTToken_, TTTrackGTTHandle);
    iEvent.getByToken(ttTrackSelectedToken_, TTTrackSelectedHandle);
    iEvent.getByToken(ttTrackSelectedEmulationToken_, TTTrackSelectedEmulationHandle);
    iEvent.getByToken(ttTrackSelectedAssociatedToken_, TTTrackSelectedAssociatedHandle);
    iEvent.getByToken(ttTrackSelectedAssociatedEmulationToken_, TTTrackSelectedAssociatedEmulationHandle);
    iEvent.getByToken(ttTrackSelectedForJetsToken_, TTTrackSelectedForJetsHandle);
    iEvent.getByToken(ttTrackSelectedEmulationForJetsToken_, TTTrackSelectedEmulationForJetsHandle);
    iEvent.getByToken(ttTrackSelectedAssociatedForJetsToken_, TTTrackSelectedAssociatedForJetsHandle);
    iEvent.getByToken(ttTrackSelectedAssociatedEmulationForJetsToken_, TTTrackSelectedAssociatedEmulationForJetsHandle);
    iEvent.getByToken(ttTrackSelectedForEtMissToken_, TTTrackSelectedForEtMissHandle);
    iEvent.getByToken(ttTrackSelectedEmulationForEtMissToken_, TTTrackSelectedEmulationForEtMissHandle);
    iEvent.getByToken(ttTrackSelectedAssociatedForEtMissToken_, TTTrackSelectedAssociatedForEtMissHandle);
    iEvent.getByToken(ttTrackSelectedAssociatedEmulationForEtMissToken_,
                      TTTrackSelectedAssociatedEmulationForEtMissHandle);
  }
  if (Displaced == "Displaced" || Displaced == "Both") {
    iEvent.getByToken(TrackFastJetsExtendedToken_, TrackFastJetsExtendedHandle);
    iEvent.getByToken(TrackJetsExtendedToken_, TrackJetsExtendedHandle);
    iEvent.getByToken(TrackJetsExtendedEmuToken_, TrackJetsExtendedEmuHandle);
    iEvent.getByToken(TrackMETExtendedToken_, L1TkMETExtendedHandle);
    iEvent.getByToken(TrackMHTExtendedToken_, L1TkMHTExtendedHandle);
    iEvent.getByToken(TrackMHTEmuExtendedToken_, L1TkMHTEmuExtendedHandle);
    iEvent.getByToken(ttTrackExtendedToken_, TTTrackExtendedHandle);
    iEvent.getByToken(ttTrackMCTruthExtendedToken_, MCTruthTTTrackExtendedHandle);
    iEvent.getByToken(ttTrackExtendedGTTToken_, TTTrackExtendedGTTHandle);
    iEvent.getByToken(ttTrackExtendedSelectedToken_, TTTrackExtendedSelectedHandle);
    iEvent.getByToken(ttTrackExtendedSelectedEmulationToken_, TTTrackExtendedSelectedEmulationHandle);
    iEvent.getByToken(ttTrackExtendedSelectedAssociatedToken_, TTTrackExtendedSelectedAssociatedHandle);
    iEvent.getByToken(ttTrackExtendedSelectedAssociatedEmulationToken_,
                      TTTrackExtendedSelectedAssociatedEmulationHandle);
    iEvent.getByToken(ttTrackExtendedSelectedForJetsToken_, TTTrackExtendedSelectedForJetsHandle);
    iEvent.getByToken(ttTrackExtendedSelectedEmulationForJetsToken_, TTTrackExtendedSelectedEmulationForJetsHandle);
    iEvent.getByToken(ttTrackExtendedSelectedAssociatedForJetsToken_, TTTrackExtendedSelectedAssociatedForJetsHandle);
    iEvent.getByToken(ttTrackExtendedSelectedAssociatedEmulationForJetsToken_,
                      TTTrackExtendedSelectedAssociatedEmulationForJetsHandle);
    iEvent.getByToken(ttTrackExtendedSelectedForEtMissToken_, TTTrackExtendedSelectedForEtMissHandle);
    iEvent.getByToken(ttTrackExtendedSelectedEmulationForEtMissToken_, TTTrackExtendedSelectedEmulationForEtMissHandle);
    iEvent.getByToken(ttTrackExtendedSelectedAssociatedForEtMissToken_,
                      TTTrackExtendedSelectedAssociatedForEtMissHandle);
    iEvent.getByToken(ttTrackExtendedSelectedAssociatedEmulationForEtMissToken_,
                      TTTrackExtendedSelectedAssociatedEmulationForEtMissHandle);
  }

  //Loop over gen particles
  if (GenParticleHandle.isValid()) {
    vector<reco::GenParticle>::const_iterator genpartIter;

    float zvtx_gen = -999;
    float trueMETx = 0;
    float trueMETy = 0;
    trueMET = 0;
    for (genpartIter = GenParticleHandle->begin(); genpartIter != GenParticleHandle->end(); ++genpartIter) {
      int status = genpartIter->status();
      if (status != 1)
        continue;
      zvtx_gen = genpartIter->vz();  //for gen vertex
      int id = genpartIter->pdgId();
      bool isNeutrino = false;
      if ((std::abs(id) == 12 || std::abs(id) == 14 || std::abs(id) == 16))
        isNeutrino = true;
      if (isNeutrino || id == 1000022) {
        trueMETx += genpartIter->pt() * cos(genpartIter->phi());
        trueMETy += genpartIter->pt() * sin(genpartIter->phi());
      }

      m_gen_pt->push_back(genpartIter->pt());
      m_gen_phi->push_back(genpartIter->phi());
      m_gen_pdgid->push_back(genpartIter->pdgId());
      m_gen_z0->push_back(zvtx_gen);
    }

    trueMET = sqrt(trueMETx * trueMETx + trueMETy * trueMETy);
    m_pv_MC->push_back(zvtx_gen);
  } else {
    edm::LogWarning("DataNotFound") << "\nWarning: GenParticleHandle not found in the event" << std::endl;
  }

  // ----------------------------------------------------------------------------------------------
  // loop over L1 stubs
  // ----------------------------------------------------------------------------------------------
  if (SaveStubs) {
    for (auto gd = tGeom.dets().begin(); gd != tGeom.dets().end(); gd++) {
      DetId detid = (*gd)->geographicalId();
      if (detid.subdetId() != StripSubdetector::TOB && detid.subdetId() != StripSubdetector::TID)
        continue;
      if (!tTopo.isLower(detid))
        continue;                             // loop on the stacks: choose the lower arbitrarily
      DetId stackDetid = tTopo.stack(detid);  // Stub module detid

      if (TTStubHandle->find(stackDetid) == TTStubHandle->end())
        continue;

      // Get the DetSets of the Clusters
      edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_>> stubs = (*TTStubHandle)[stackDetid];
      const GeomDetUnit* det0 = tGeom.idToDetUnit(detid);
      const auto* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(det0);
      const PixelTopology* topol = dynamic_cast<const PixelTopology*>(&(theGeomDet->specificTopology()));

      // loop over stubs
      for (auto stubIter = stubs.begin(); stubIter != stubs.end(); ++stubIter) {
        edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>> tempStubPtr =
            edmNew::makeRefTo(TTStubHandle, stubIter);

        int isBarrel = 0;
        int layer = -999999;
        if (detid.subdetId() == StripSubdetector::TOB) {
          isBarrel = 1;
          layer = static_cast<int>(tTopo.layer(detid));
        } else if (detid.subdetId() == StripSubdetector::TID) {
          isBarrel = 0;
          layer = static_cast<int>(tTopo.layer(detid));
        } else {
          edm::LogVerbatim("Tracklet") << "WARNING -- neither TOB or TID stub, shouldn't happen...";
          layer = -1;
        }

        int isPSmodule = 0;
        if (topol->nrows() == 960)
          isPSmodule = 1;

        MeasurementPoint coords = tempStubPtr->clusterRef(0)->findAverageLocalCoordinatesCentered();
        LocalPoint clustlp = topol->localPosition(coords);
        GlobalPoint posStub = theGeomDet->surface().toGlobal(clustlp);

        double tmp_stub_x = posStub.x();
        double tmp_stub_y = posStub.y();
        double tmp_stub_z = posStub.z();

        float trigDisplace = tempStubPtr->rawBend();
        float trigOffset = tempStubPtr->bendOffset();
        float trigPos = tempStubPtr->innerClusterPosition();
        float trigBend = tempStubPtr->bendFE();

        m_allstub_x->push_back(tmp_stub_x);
        m_allstub_y->push_back(tmp_stub_y);
        m_allstub_z->push_back(tmp_stub_z);
        m_allstub_isBarrel->push_back(isBarrel);
        m_allstub_layer->push_back(layer);
        m_allstub_isPSmodule->push_back(isPSmodule);
        m_allstub_trigDisplace->push_back(trigDisplace);
        m_allstub_trigOffset->push_back(trigOffset);
        m_allstub_trigPos->push_back(trigPos);
        m_allstub_trigBend->push_back(trigBend);

        // matched to tracking particle?
        edm::Ptr<TrackingParticle> my_tp = MCTruthTTStubHandle->findTrackingParticlePtr(tempStubPtr);

        int myTP_pdgid = -999;
        float myTP_pt = -999;
        float myTP_eta = -999;
        float myTP_phi = -999;

        if (my_tp.isNull() == false) {
          int tmp_eventid = my_tp->eventId().event();
          if (tmp_eventid > 0)
            continue;  // this means stub from pileup track
          myTP_pdgid = my_tp->pdgId();
          myTP_pt = my_tp->p4().pt();
          myTP_eta = my_tp->p4().eta();
          myTP_phi = my_tp->p4().phi();
        }
        int tmp_stub_genuine = 0;
        if (MCTruthTTStubHandle->isGenuine(tempStubPtr))
          tmp_stub_genuine = 1;

        m_allstub_matchTP_pdgid->push_back(myTP_pdgid);
        m_allstub_matchTP_pt->push_back(myTP_pt);
        m_allstub_matchTP_eta->push_back(myTP_eta);
        m_allstub_matchTP_phi->push_back(myTP_phi);
        m_allstub_genuine->push_back(tmp_stub_genuine);
      }
    }
  }

  // ----------------------------------------------------------------------------------------------
  // loop over (prompt) L1 tracks
  // ----------------------------------------------------------------------------------------------
  if (SaveAllTracks && (Displaced == "Prompt" || Displaced == "Both")) {
    if (DebugMode) {
      edm::LogVerbatim("Tracklet") << "\n Loop over L1 tracks!";
      edm::LogVerbatim("Tracklet") << "\n Looking at " << Displaced << " tracks!";
    }

    int this_l1track = 0;
    for (iterL1Track = TTTrackHandle->begin(); iterL1Track != TTTrackHandle->end(); iterL1Track++) {
      L1TrackPtr l1track_ptr(TTTrackHandle, this_l1track);
      L1TrackRef l1track_ref(TTTrackGTTHandle, this_l1track);
      this_l1track++;

      float tmp_trk_pt = iterL1Track->momentum().perp();
      float tmp_trk_eta = iterL1Track->momentum().eta();
      float tmp_trk_phi = iterL1Track->momentum().phi();
      float tmp_trk_phi_local = iterL1Track->localPhi();
      float tmp_trk_z0 = iterL1Track->z0();            //cm
      int tmp_trk_nFitPars = iterL1Track->nFitPars();  //4 or 5

      float tmp_trk_d0 = -999;
      if (tmp_trk_nFitPars == 5) {
        float tmp_trk_x0 = iterL1Track->POCA().x();
        float tmp_trk_y0 = iterL1Track->POCA().y();
        tmp_trk_d0 = -tmp_trk_x0 * sin(tmp_trk_phi) + tmp_trk_y0 * cos(tmp_trk_phi);
        // tmp_trk_d0 = iterL1Track->d0();
      }

      float tmp_trk_chi2 = iterL1Track->chi2();
      float tmp_trk_chi2dof = iterL1Track->chi2Red();
      float tmp_trk_chi2rphi = iterL1Track->chi2XYRed();
      float tmp_trk_chi2rz = iterL1Track->chi2ZRed();
      float tmp_trk_bendchi2 = iterL1Track->stubPtConsistency();
      float tmp_trk_MVA1 = iterL1Track->trkMVA1();

      std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
          stubRefs = iterL1Track->getStubRefs();
      int tmp_trk_nstub = (int)stubRefs.size();
      int tmp_trk_seed = 0;
      tmp_trk_seed = (int)iterL1Track->trackSeedType();
      int tmp_trk_hitpattern = 0;
      tmp_trk_hitpattern = (int)iterL1Track->hitPattern();
      unsigned int tmp_trk_phiSector = iterL1Track->phiSector();

      // ----------------------------------------------------------------------------------------------
      // loop over stubs on tracks
      int tmp_trk_dhits = 0;
      int tmp_trk_lhits = 0;
      if (true) {
        // loop over stubs
        for (int is = 0; is < tmp_trk_nstub; is++) {
          //detID of stub
          DetId detIdStub = tGeom.idToDet((stubRefs.at(is)->clusterRef(0))->getDetId())->geographicalId();
          MeasurementPoint coords = stubRefs.at(is)->clusterRef(0)->findAverageLocalCoordinatesCentered();
          const GeomDet* theGeomDet = tGeom.idToDet(detIdStub);
          Global3DPoint posStub = theGeomDet->surface().toGlobal(theGeomDet->topology().localPosition(coords));

          double x = posStub.x();
          double y = posStub.y();
          double z = posStub.z();

          int layer = -999999;
          if (detIdStub.subdetId() == StripSubdetector::TOB) {
            layer = static_cast<int>(tTopo.layer(detIdStub));
            if (DebugMode)
              edm::LogVerbatim("Tracklet")
                  << "   stub in layer " << layer << " at position x y z = " << x << " " << y << " " << z;
            tmp_trk_lhits += pow(10, layer - 1);
          } else if (detIdStub.subdetId() == StripSubdetector::TID) {
            layer = static_cast<int>(tTopo.layer(detIdStub));
            if (DebugMode)
              edm::LogVerbatim("Tracklet")
                  << "   stub in disk " << layer << " at position x y z = " << x << " " << y << " " << z;
            tmp_trk_dhits += pow(10, layer - 1);
          }
        }  //end loop over stubs
      }
      // ----------------------------------------------------------------------------------------------

      int tmp_trk_genuine = 0;
      int tmp_trk_loose = 0;
      int tmp_trk_unknown = 0;
      int tmp_trk_combinatoric = 0;
      if (MCTruthTTTrackHandle->isLooselyGenuine(l1track_ptr))
        tmp_trk_loose = 1;
      if (MCTruthTTTrackHandle->isGenuine(l1track_ptr))
        tmp_trk_genuine = 1;
      if (MCTruthTTTrackHandle->isUnknown(l1track_ptr))
        tmp_trk_unknown = 1;
      if (MCTruthTTTrackHandle->isCombinatoric(l1track_ptr))
        tmp_trk_combinatoric = 1;

      if (DebugMode) {
        edm::LogVerbatim("Tracklet") << "L1 track,"
                                     << " pt: " << tmp_trk_pt << " eta: " << tmp_trk_eta << " phi: " << tmp_trk_phi
                                     << " z0: " << tmp_trk_z0 << " chi2: " << tmp_trk_chi2
                                     << " chi2rphi: " << tmp_trk_chi2rphi << " chi2rz: " << tmp_trk_chi2rz
                                     << " nstub: " << tmp_trk_nstub;
        if (tmp_trk_genuine)
          edm::LogVerbatim("Tracklet") << "    (is genuine)";
        if (tmp_trk_unknown)
          edm::LogVerbatim("Tracklet") << "    (is unknown)";
        if (tmp_trk_combinatoric)
          edm::LogVerbatim("Tracklet") << "    (is combinatoric)";
      }

      m_trk_pt->push_back(tmp_trk_pt);
      m_trk_eta->push_back(tmp_trk_eta);
      m_trk_phi->push_back(tmp_trk_phi);
      m_trk_phi_local->push_back(tmp_trk_phi_local);
      m_trk_z0->push_back(tmp_trk_z0);
      if (tmp_trk_nFitPars == 5)
        m_trk_d0->push_back(tmp_trk_d0);
      else
        m_trk_d0->push_back(999.);
      m_trk_chi2->push_back(tmp_trk_chi2);
      m_trk_chi2dof->push_back(tmp_trk_chi2dof);
      m_trk_chi2rphi->push_back(tmp_trk_chi2rphi);
      m_trk_chi2rz->push_back(tmp_trk_chi2rz);
      m_trk_bendchi2->push_back(tmp_trk_bendchi2);
      m_trk_MVA1->push_back(tmp_trk_MVA1);
      m_trk_nstub->push_back(tmp_trk_nstub);
      m_trk_dhits->push_back(tmp_trk_dhits);
      m_trk_lhits->push_back(tmp_trk_lhits);
      m_trk_seed->push_back(tmp_trk_seed);
      m_trk_hitpattern->push_back(tmp_trk_hitpattern);
      m_trk_phiSector->push_back(tmp_trk_phiSector);
      m_trk_genuine->push_back(tmp_trk_genuine);
      m_trk_loose->push_back(tmp_trk_loose);
      m_trk_unknown->push_back(tmp_trk_unknown);
      m_trk_combinatoric->push_back(tmp_trk_combinatoric);

      // ----------------------------------------------------------------------------------------------
      // for studying the fake rate
      // ----------------------------------------------------------------------------------------------
      edm::Ptr<TrackingParticle> my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(l1track_ptr);

      int myFake = 0;
      int myTP_pdgid = -999;
      float myTP_pt = -999;
      float myTP_eta = -999;
      float myTP_phi = -999;
      float myTP_z0 = -999;
      float myTP_dxy = -999;

      if (my_tp.isNull())
        myFake = 0;
      else {
        int tmp_eventid = my_tp->eventId().event();
        if (tmp_eventid > 0)
          myFake = 2;
        else
          myFake = 1;

        myTP_pdgid = my_tp->pdgId();
        myTP_pt = my_tp->p4().pt();
        myTP_eta = my_tp->p4().eta();
        myTP_phi = my_tp->p4().phi();
        myTP_z0 = my_tp->vertex().z();

        float myTP_x0 = my_tp->vertex().x();
        float myTP_y0 = my_tp->vertex().y();
        myTP_dxy = sqrt(myTP_x0 * myTP_x0 + myTP_y0 * myTP_y0);

        if (DebugMode) {
          edm::LogVerbatim("Tracklet") << "TP matched to track has pt = " << my_tp->p4().pt()
                                       << " eta = " << my_tp->momentum().eta() << " phi = " << my_tp->momentum().phi()
                                       << " z0 = " << my_tp->vertex().z() << " pdgid = " << my_tp->pdgId()
                                       << " dxy = " << myTP_dxy;
        }
      }

      m_trk_fake->push_back(myFake);
      m_trk_matchtp_pdgid->push_back(myTP_pdgid);
      m_trk_matchtp_pt->push_back(myTP_pt);
      m_trk_matchtp_eta->push_back(myTP_eta);
      m_trk_matchtp_phi->push_back(myTP_phi);
      m_trk_matchtp_z0->push_back(myTP_z0);
      m_trk_matchtp_dxy->push_back(myTP_dxy);

      // ----------------------------------------------------------------------------------------------
      // store the index to the selected track or -1 if not selected
      // ----------------------------------------------------------------------------------------------
      m_trk_gtt_pt->push_back(l1track_ref->momentum().perp());
      m_trk_gtt_eta->push_back(l1track_ref->momentum().eta());
      m_trk_gtt_phi->push_back(l1track_ref->momentum().phi());
      m_trk_selected_index->push_back(getSelectedTrackIndex(l1track_ref, TTTrackSelectedHandle));
      m_trk_selected_emulation_index->push_back(getSelectedTrackIndex(l1track_ref, TTTrackSelectedEmulationHandle));
      m_trk_selected_associated_index->push_back(getSelectedTrackIndex(l1track_ref, TTTrackSelectedAssociatedHandle));
      m_trk_selected_associated_emulation_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackSelectedAssociatedEmulationHandle));
      m_trk_selected_forjets_index->push_back(getSelectedTrackIndex(l1track_ref, TTTrackSelectedForJetsHandle));
      m_trk_selected_emulation_forjets_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackSelectedEmulationForJetsHandle));
      m_trk_selected_associated_forjets_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackSelectedAssociatedForJetsHandle));
      m_trk_selected_associated_emulation_forjets_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackSelectedAssociatedEmulationForJetsHandle));
      m_trk_selected_foretmiss_index->push_back(getSelectedTrackIndex(l1track_ref, TTTrackSelectedForEtMissHandle));
      m_trk_selected_emulation_foretmiss_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackSelectedEmulationForEtMissHandle));
      m_trk_selected_associated_foretmiss_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackSelectedAssociatedForEtMissHandle));
      m_trk_selected_associated_emulation_foretmiss_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackSelectedAssociatedEmulationForEtMissHandle));
    }  //end track loop
  }    //end if SaveAllTracks

  // ----------------------------------------------------------------------------------------------
  // loop over (extended) L1 tracks
  // ----------------------------------------------------------------------------------------------
  if (SaveAllTracks && (Displaced == "Displaced" || Displaced == "Both")) {
    if (DebugMode) {
      edm::LogVerbatim("Tracklet") << "\n Loop over L1 tracks!";
      edm::LogVerbatim("Tracklet") << "\n Looking at " << Displaced << " tracks!";
    }

    int this_l1track = 0;
    for (iterL1Track = TTTrackExtendedHandle->begin(); iterL1Track != TTTrackExtendedHandle->end(); iterL1Track++) {
      L1TrackPtr l1track_ptr(TTTrackExtendedHandle, this_l1track);
      L1TrackRef l1track_ref(TTTrackExtendedGTTHandle, this_l1track);
      this_l1track++;

      float tmp_trk_pt = iterL1Track->momentum().perp();
      float tmp_trk_eta = iterL1Track->momentum().eta();
      float tmp_trk_phi = iterL1Track->momentum().phi();
      float tmp_trk_phi_local = iterL1Track->localPhi();
      float tmp_trk_z0 = iterL1Track->z0();            //cm
      int tmp_trk_nFitPars = iterL1Track->nFitPars();  //4 or 5

      float tmp_trk_d0 = -999;
      if (tmp_trk_nFitPars == 5) {
        float tmp_trk_x0 = iterL1Track->POCA().x();
        float tmp_trk_y0 = iterL1Track->POCA().y();
        tmp_trk_d0 = -tmp_trk_x0 * sin(tmp_trk_phi) + tmp_trk_y0 * cos(tmp_trk_phi);
        // tmp_trk_d0 = iterL1Track->d0();
      }

      float tmp_trk_chi2 = iterL1Track->chi2();
      float tmp_trk_chi2dof = iterL1Track->chi2Red();
      float tmp_trk_chi2rphi = iterL1Track->chi2XYRed();
      float tmp_trk_chi2rz = iterL1Track->chi2ZRed();
      float tmp_trk_bendchi2 = iterL1Track->stubPtConsistency();
      float tmp_trk_MVA1 = iterL1Track->trkMVA1();

      std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
          stubRefs = iterL1Track->getStubRefs();
      int tmp_trk_nstub = (int)stubRefs.size();
      int tmp_trk_seed = 0;
      tmp_trk_seed = (int)iterL1Track->trackSeedType();
      int tmp_trk_hitpattern = 0;
      tmp_trk_hitpattern = (int)iterL1Track->hitPattern();
      unsigned int tmp_trk_phiSector = iterL1Track->phiSector();

      // ----------------------------------------------------------------------------------------------
      // loop over stubs on tracks
      int tmp_trk_dhits = 0;
      int tmp_trk_lhits = 0;
      if (true) {
        // loop over stubs
        for (int is = 0; is < tmp_trk_nstub; is++) {
          //detID of stub
          DetId detIdStub = tGeom.idToDet((stubRefs.at(is)->clusterRef(0))->getDetId())->geographicalId();
          MeasurementPoint coords = stubRefs.at(is)->clusterRef(0)->findAverageLocalCoordinatesCentered();
          const GeomDet* theGeomDet = tGeom.idToDet(detIdStub);
          Global3DPoint posStub = theGeomDet->surface().toGlobal(theGeomDet->topology().localPosition(coords));

          double x = posStub.x();
          double y = posStub.y();
          double z = posStub.z();

          int layer = -999999;
          if (detIdStub.subdetId() == StripSubdetector::TOB) {
            layer = static_cast<int>(tTopo.layer(detIdStub));
            if (DebugMode)
              edm::LogVerbatim("Tracklet")
                  << "   stub in layer " << layer << " at position x y z = " << x << " " << y << " " << z;
            tmp_trk_lhits += pow(10, layer - 1);
          } else if (detIdStub.subdetId() == StripSubdetector::TID) {
            layer = static_cast<int>(tTopo.layer(detIdStub));
            if (DebugMode)
              edm::LogVerbatim("Tracklet")
                  << "   stub in disk " << layer << " at position x y z = " << x << " " << y << " " << z;
            tmp_trk_dhits += pow(10, layer - 1);
          }
        }  //end loop over stubs
      }
      // ----------------------------------------------------------------------------------------------

      int tmp_trk_genuine = 0;
      int tmp_trk_loose = 0;
      int tmp_trk_unknown = 0;
      int tmp_trk_combinatoric = 0;
      if (MCTruthTTTrackExtendedHandle->isLooselyGenuine(l1track_ptr))
        tmp_trk_loose = 1;
      if (MCTruthTTTrackExtendedHandle->isGenuine(l1track_ptr))
        tmp_trk_genuine = 1;
      if (MCTruthTTTrackExtendedHandle->isUnknown(l1track_ptr))
        tmp_trk_unknown = 1;
      if (MCTruthTTTrackExtendedHandle->isCombinatoric(l1track_ptr))
        tmp_trk_combinatoric = 1;

      if (DebugMode) {
        edm::LogVerbatim("Tracklet") << "L1 track,"
                                     << " pt: " << tmp_trk_pt << " eta: " << tmp_trk_eta << " phi: " << tmp_trk_phi
                                     << " z0: " << tmp_trk_z0 << " chi2: " << tmp_trk_chi2
                                     << " chi2rphi: " << tmp_trk_chi2rphi << " chi2rz: " << tmp_trk_chi2rz
                                     << " nstub: " << tmp_trk_nstub;
        if (tmp_trk_genuine)
          edm::LogVerbatim("Tracklet") << "    (is genuine)";
        if (tmp_trk_unknown)
          edm::LogVerbatim("Tracklet") << "    (is unknown)";
        if (tmp_trk_combinatoric)
          edm::LogVerbatim("Tracklet") << "    (is combinatoric)";
      }

      m_trkExt_pt->push_back(tmp_trk_pt);
      m_trkExt_eta->push_back(tmp_trk_eta);
      m_trkExt_phi->push_back(tmp_trk_phi);
      m_trkExt_phi_local->push_back(tmp_trk_phi_local);
      m_trkExt_z0->push_back(tmp_trk_z0);
      if (tmp_trk_nFitPars == 5)
        m_trkExt_d0->push_back(tmp_trk_d0);
      else
        m_trkExt_d0->push_back(999.);
      m_trkExt_chi2->push_back(tmp_trk_chi2);
      m_trkExt_chi2dof->push_back(tmp_trk_chi2dof);
      m_trkExt_chi2rphi->push_back(tmp_trk_chi2rphi);
      m_trkExt_chi2rz->push_back(tmp_trk_chi2rz);
      m_trkExt_bendchi2->push_back(tmp_trk_bendchi2);
      m_trkExt_MVA->push_back(tmp_trk_MVA1);
      m_trkExt_nstub->push_back(tmp_trk_nstub);
      m_trkExt_dhits->push_back(tmp_trk_dhits);
      m_trkExt_lhits->push_back(tmp_trk_lhits);
      m_trkExt_seed->push_back(tmp_trk_seed);
      m_trkExt_hitpattern->push_back(tmp_trk_hitpattern);
      m_trkExt_phiSector->push_back(tmp_trk_phiSector);
      m_trkExt_genuine->push_back(tmp_trk_genuine);
      m_trkExt_loose->push_back(tmp_trk_loose);
      m_trkExt_unknown->push_back(tmp_trk_unknown);
      m_trkExt_combinatoric->push_back(tmp_trk_combinatoric);

      // ----------------------------------------------------------------------------------------------
      // for studying the fake rate
      // ----------------------------------------------------------------------------------------------
      edm::Ptr<TrackingParticle> my_tp = MCTruthTTTrackExtendedHandle->findTrackingParticlePtr(l1track_ptr);

      int myFake = 0;
      int myTP_pdgid = -999;
      float myTP_pt = -999;
      float myTP_eta = -999;
      float myTP_phi = -999;
      float myTP_z0 = -999;
      float myTP_dxy = -999;

      if (my_tp.isNull())
        myFake = 0;
      else {
        int tmp_eventid = my_tp->eventId().event();
        if (tmp_eventid > 0)
          myFake = 2;
        else
          myFake = 1;

        myTP_pdgid = my_tp->pdgId();
        myTP_pt = my_tp->p4().pt();
        myTP_eta = my_tp->p4().eta();
        myTP_phi = my_tp->p4().phi();
        myTP_z0 = my_tp->vertex().z();

        float myTP_x0 = my_tp->vertex().x();
        float myTP_y0 = my_tp->vertex().y();
        myTP_dxy = sqrt(myTP_x0 * myTP_x0 + myTP_y0 * myTP_y0);

        if (DebugMode) {
          edm::LogVerbatim("Tracklet") << "TP matched to track has pt = " << my_tp->p4().pt()
                                       << " eta = " << my_tp->momentum().eta() << " phi = " << my_tp->momentum().phi()
                                       << " z0 = " << my_tp->vertex().z() << " pdgid = " << my_tp->pdgId()
                                       << " dxy = " << myTP_dxy;
        }
      }

      m_trkExt_fake->push_back(myFake);
      m_trkExt_matchtp_pdgid->push_back(myTP_pdgid);
      m_trkExt_matchtp_pt->push_back(myTP_pt);
      m_trkExt_matchtp_eta->push_back(myTP_eta);
      m_trkExt_matchtp_phi->push_back(myTP_phi);
      m_trkExt_matchtp_z0->push_back(myTP_z0);
      m_trkExt_matchtp_dxy->push_back(myTP_dxy);

      // ----------------------------------------------------------------------------------------------
      // store the index to the selected track or -1 if not selected
      // ----------------------------------------------------------------------------------------------
      m_trkExt_gtt_pt->push_back(l1track_ref->momentum().perp());
      m_trkExt_gtt_eta->push_back(l1track_ref->momentum().eta());
      m_trkExt_gtt_phi->push_back(l1track_ref->momentum().phi());
      m_trkExt_selected_index->push_back(getSelectedTrackIndex(l1track_ref, TTTrackExtendedSelectedHandle));
      m_trkExt_selected_emulation_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackExtendedSelectedEmulationHandle));
      m_trkExt_selected_associated_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackExtendedSelectedAssociatedHandle));
      m_trkExt_selected_associated_emulation_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackExtendedSelectedAssociatedEmulationHandle));
      m_trkExt_selected_forjets_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackExtendedSelectedForJetsHandle));
      m_trkExt_selected_emulation_forjets_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackExtendedSelectedEmulationForJetsHandle));
      m_trkExt_selected_associated_forjets_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackExtendedSelectedAssociatedForJetsHandle));
      m_trkExt_selected_associated_emulation_forjets_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackExtendedSelectedAssociatedEmulationForJetsHandle));
      m_trkExt_selected_foretmiss_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackExtendedSelectedForEtMissHandle));
      m_trkExt_selected_emulation_foretmiss_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackExtendedSelectedEmulationForEtMissHandle));
      m_trkExt_selected_associated_foretmiss_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackExtendedSelectedAssociatedForEtMissHandle));
      m_trkExt_selected_associated_emulation_foretmiss_index->push_back(
          getSelectedTrackIndex(l1track_ref, TTTrackExtendedSelectedAssociatedEmulationForEtMissHandle));
    }  //end track loop
  }    //end if SaveAllTracks (displaced)

  // ----------------------------------------------------------------------------------------------
  // loop over tracking particles
  // ----------------------------------------------------------------------------------------------
  if (DebugMode)
    edm::LogVerbatim("Tracklet") << "\n Loop over tracking particles!";

  trueTkMET = 0;
  float trueTkMETx = 0;
  float trueTkMETy = 0;

  int this_tp = 0;
  std::vector<TrackingParticle>::const_iterator iterTP;
  for (iterTP = TrackingParticleHandle->begin(); iterTP != TrackingParticleHandle->end(); ++iterTP) {
    edm::Ptr<TrackingParticle> tp_ptr(TrackingParticleHandle, this_tp);
    this_tp++;

    int tmp_eventid = iterTP->eventId().event();
    if (MyProcess != 1 && tmp_eventid > 0)
      continue;  //only care about primary interaction

    float tmp_tp_pt = iterTP->pt();
    float tmp_tp_eta = iterTP->eta();
    float tmp_tp_phi = iterTP->phi();
    float tmp_tp_vz = iterTP->vz();
    float tmp_tp_vx = iterTP->vx();
    float tmp_tp_vy = iterTP->vy();
    int tmp_tp_pdgid = iterTP->pdgId();
    float tmp_tp_z0_prod = tmp_tp_vz;
    float tmp_tp_d0_prod = tmp_tp_vx * sin(tmp_tp_phi) - tmp_tp_vy * cos(tmp_tp_phi);

    if (MyProcess == 13 && abs(tmp_tp_pdgid) != 13)
      continue;
    if (MyProcess == 11 && abs(tmp_tp_pdgid) != 11)
      continue;
    if ((MyProcess == 6 || MyProcess == 15 || MyProcess == 211) && abs(tmp_tp_pdgid) != 211)
      continue;

    if (tmp_tp_pt < TP_minPt)
      continue;
    if (std::abs(tmp_tp_eta) > TP_maxEta)
      continue;

    // ----------------------------------------------------------------------------------------------
    // get d0/z0 propagated back to the IP
    float tmp_tp_t = tan(2.0 * atan(1.0) - 2.0 * atan(exp(-tmp_tp_eta)));
    float delx = -tmp_tp_vx;
    float dely = -tmp_tp_vy;

    float A = 0.01 * 0.5696;
    float Kmagnitude = A / tmp_tp_pt;
    float tmp_tp_charge = tp_ptr->charge();
    float K = Kmagnitude * tmp_tp_charge;
    float d = 0;
    float tmp_tp_x0p = delx - (d + 1. / (2. * K) * sin(tmp_tp_phi));
    float tmp_tp_y0p = dely + (d + 1. / (2. * K) * cos(tmp_tp_phi));
    float tmp_tp_rp = sqrt(tmp_tp_x0p * tmp_tp_x0p + tmp_tp_y0p * tmp_tp_y0p);
    float tmp_tp_d0 = tmp_tp_charge * tmp_tp_rp - (1. / (2. * K));
    tmp_tp_d0 = tmp_tp_d0 * (-1);  //fix d0 sign
    const double pi = 4.0 * atan(1.0);
    float delphi = tmp_tp_phi - atan2(-K * tmp_tp_x0p, K * tmp_tp_y0p);
    if (delphi < -pi)
      delphi += 2.0 * pi;
    if (delphi > pi)
      delphi -= 2.0 * pi;
    float tmp_tp_z0 = tmp_tp_vz + tmp_tp_t * delphi / (2.0 * K);
    // ----------------------------------------------------------------------------------------------

    if (std::abs(tmp_tp_z0) > TP_maxZ0)
      continue;

    // for pions in ttbar, only consider TPs coming from near the IP!
    float dxy = sqrt(tmp_tp_vx * tmp_tp_vx + tmp_tp_vy * tmp_tp_vy);
    float tmp_tp_dxy = dxy;
    if (MyProcess == 6 && (dxy > 1.0))
      continue;

    if (DebugMode && (Displaced == "Prompt" || Displaced == "Both"))
      edm::LogVerbatim("Tracklet") << "Tracking particle, pt: " << tmp_tp_pt << " eta: " << tmp_tp_eta
                                   << " phi: " << tmp_tp_phi << " z0: " << tmp_tp_z0 << " d0: " << tmp_tp_d0
                                   << " z_prod: " << tmp_tp_z0_prod << " d_prod: " << tmp_tp_d0_prod
                                   << " pdgid: " << tmp_tp_pdgid << " eventID: " << iterTP->eventId().event()
                                   << " ttclusters " << MCTruthTTClusterHandle->findTTClusterRefs(tp_ptr).size()
                                   << " ttstubs " << MCTruthTTStubHandle->findTTStubRefs(tp_ptr).size() << " tttracks "
                                   << MCTruthTTTrackHandle->findTTTrackPtrs(tp_ptr).size();

    // ----------------------------------------------------------------------------------------------
    // only consider TPs associated with >= 1 cluster, or >= X stubs, or have stubs in >= X layers (configurable options)
    if (MCTruthTTClusterHandle->findTTClusterRefs(tp_ptr).empty()) {
      if (DebugMode)
        edm::LogVerbatim("Tracklet") << "No matching TTClusters for TP, continuing...";
      continue;
    }

    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
        theStubRefs = MCTruthTTStubHandle->findTTStubRefs(tp_ptr);
    int nStubTP = (int)theStubRefs.size();

    // how many layers/disks have stubs?
    int hasStubInLayer[11] = {0};
    for (auto& theStubRef : theStubRefs) {
      DetId detid(theStubRef->getDetId());

      int layer = -1;
      if (detid.subdetId() == StripSubdetector::TOB) {
        layer = static_cast<int>(tTopo.layer(detid)) - 1;  //fill in array as entries 0-5
      } else if (detid.subdetId() == StripSubdetector::TID) {
        layer = static_cast<int>(tTopo.layer(detid)) + 5;  //fill in array as entries 6-10
      }

      //treat genuine stubs separately (==2 is genuine, ==1 is not)
      if (MCTruthTTStubHandle->findTrackingParticlePtr(theStubRef).isNull() && hasStubInLayer[layer] < 2)
        hasStubInLayer[layer] = 1;
      else
        hasStubInLayer[layer] = 2;
    }

    int nStubLayerTP = 0;
    int nStubLayerTP_g = 0;
    for (int isum : hasStubInLayer) {
      if (isum >= 1)
        nStubLayerTP += 1;
      if (isum == 2)
        nStubLayerTP_g += 1;
    }

    if (DebugMode)
      edm::LogVerbatim("Tracklet") << "TP is associated with " << nStubTP << " stubs, and has stubs in " << nStubLayerTP
                                   << " different layers/disks, and has GENUINE stubs in " << nStubLayerTP_g
                                   << " layers ";

    if (TP_minNStub > 0) {
      if (DebugMode)
        edm::LogVerbatim("Tracklet") << "Only consider TPs with >= " << TP_minNStub << " stubs";
      if (nStubTP < TP_minNStub) {
        if (DebugMode)
          edm::LogVerbatim("Tracklet") << "TP fails minimum nbr stubs requirement! Continuing...";
        continue;
      }
    }
    if (TP_minNStubLayer > 0) {
      if (DebugMode)
        edm::LogVerbatim("Tracklet") << "Only consider TPs with stubs in >= " << TP_minNStubLayer << " layers/disks";
      if (nStubLayerTP < TP_minNStubLayer) {
        if (DebugMode)
          edm::LogVerbatim("Tracklet") << "TP fails stubs in minimum nbr of layers/disks requirement! Continuing...";
        continue;
      }
    }

    if (tmp_eventid == 0) {
      trueTkMETx += tmp_tp_pt * cos(tmp_tp_phi);
      trueTkMETy += tmp_tp_pt * sin(tmp_tp_phi);
    }

    m_tp_pt->push_back(tmp_tp_pt);
    m_tp_eta->push_back(tmp_tp_eta);
    m_tp_phi->push_back(tmp_tp_phi);
    m_tp_dxy->push_back(tmp_tp_dxy);
    m_tp_z0->push_back(tmp_tp_z0);
    m_tp_d0->push_back(tmp_tp_d0);
    m_tp_z0_prod->push_back(tmp_tp_z0_prod);
    m_tp_d0_prod->push_back(tmp_tp_d0_prod);
    m_tp_pdgid->push_back(tmp_tp_pdgid);
    m_tp_nstub->push_back(nStubTP);
    m_tp_eventid->push_back(tmp_eventid);
    m_tp_charge->push_back(tmp_tp_charge);

    // ----------------------------------------------------------------------------------------------
    // look for L1 tracks (prompt) matched to the tracking particle
    if (Displaced == "Prompt" || Displaced == "Both") {
      std::vector<edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>>> matchedTracks =
          MCTruthTTTrackHandle->findTTTrackPtrs(tp_ptr);

      int nMatch = 0;
      int i_track = -1;
      float i_chi2dof = 99999;

      if (!matchedTracks.empty()) {
        if (DebugMode && (matchedTracks.size() > 1))
          edm::LogVerbatim("Tracklet") << "TrackingParticle has more than one matched L1 track!";

        // ----------------------------------------------------------------------------------------------
        // loop over matched L1 tracks
        // here, "match" means tracks that can be associated to a TrackingParticle with at least one hit of at least one of its clusters
        // https://twiki.cern.ch/twiki/bin/viewauth/CMS/SLHCTrackerTriggerSWTools#MC_truth_for_TTTrack

        for (int it = 0; it < (int)matchedTracks.size(); it++) {
          bool tmp_trk_genuine = false;
          bool tmp_trk_loosegenuine = false;
          if (MCTruthTTTrackHandle->isGenuine(matchedTracks.at(it)))
            tmp_trk_genuine = true;
          if (MCTruthTTTrackHandle->isLooselyGenuine(matchedTracks.at(it)))
            tmp_trk_loosegenuine = true;
          if (!tmp_trk_loosegenuine)
            continue;

          if (DebugMode) {
            if (MCTruthTTTrackHandle->findTrackingParticlePtr(matchedTracks.at(it)).isNull()) {
              edm::LogVerbatim("Tracklet") << "track matched to TP is NOT uniquely matched to a TP";
            } else {
              edm::Ptr<TrackingParticle> my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(matchedTracks.at(it));
              edm::LogVerbatim("Tracklet") << "TP matched to track matched to TP ... tp pt = " << my_tp->p4().pt()
                                           << " eta = " << my_tp->momentum().eta()
                                           << " phi = " << my_tp->momentum().phi() << " z0 = " << my_tp->vertex().z();
            }
            edm::LogVerbatim("Tracklet") << "   ... matched L1 track has pt = "
                                         << matchedTracks.at(it)->momentum().perp()
                                         << " eta = " << matchedTracks.at(it)->momentum().eta()
                                         << " phi = " << matchedTracks.at(it)->momentum().phi()
                                         << " chi2 = " << matchedTracks.at(it)->chi2()
                                         << " consistency = " << matchedTracks.at(it)->stubPtConsistency()
                                         << " z0 = " << matchedTracks.at(it)->z0()
                                         << " nstub = " << matchedTracks.at(it)->getStubRefs().size();
            if (tmp_trk_genuine)
              edm::LogVerbatim("Tracklet") << "    (genuine!) ";
            if (tmp_trk_loosegenuine)
              edm::LogVerbatim("Tracklet") << "    (loose genuine!) ";
          }

          std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
              stubRefs = matchedTracks.at(it)->getStubRefs();
          int tmp_trk_nstub = stubRefs.size();

          if (tmp_trk_nstub < L1Tk_minNStub)
            continue;

          float dmatch_pt = 999;
          float dmatch_eta = 999;
          float dmatch_phi = 999;
          int match_id = 999;

          edm::Ptr<TrackingParticle> my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(matchedTracks.at(it));
          dmatch_pt = std::abs(my_tp->p4().pt() - tmp_tp_pt);
          dmatch_eta = std::abs(my_tp->p4().eta() - tmp_tp_eta);
          dmatch_phi = std::abs(my_tp->p4().phi() - tmp_tp_phi);
          match_id = my_tp->pdgId();
          float tmp_trk_chi2dof = matchedTracks.at(it)->chi2Red();

          // ensure that track is uniquely matched to the TP we are looking at!
          if (dmatch_pt < 0.1 && dmatch_eta < 0.1 && dmatch_phi < 0.1 && tmp_tp_pdgid == match_id && tmp_trk_genuine) {
            nMatch++;
            if (i_track < 0 || tmp_trk_chi2dof < i_chi2dof) {
              i_track = it;
              i_chi2dof = tmp_trk_chi2dof;
            }
          }

        }  // end loop over matched L1 tracks
      }    // end has at least 1 matched L1 track
      // ----------------------------------------------------------------------------------------------

      float tmp_matchtrk_pt = -999;
      float tmp_matchtrk_eta = -999;
      float tmp_matchtrk_phi = -999;
      float tmp_matchtrk_z0 = -999;
      float tmp_matchtrk_d0 = -999;
      float tmp_matchtrk_chi2 = -999;
      float tmp_matchtrk_chi2dof = -999;
      float tmp_matchtrk_chi2rphi = -999;
      float tmp_matchtrk_chi2rz = -999;
      float tmp_matchtrk_bendchi2 = -999;
      float tmp_matchtrk_MVA1 = -999;
      int tmp_matchtrk_nstub = -999;
      int tmp_matchtrk_dhits = -999;
      int tmp_matchtrk_lhits = -999;
      int tmp_matchtrk_seed = -999;
      int tmp_matchtrk_hitpattern = -999;
      int tmp_matchtrk_nFitPars = -999;

      if (nMatch > 1 && DebugMode)
        edm::LogVerbatim("Tracklet") << "WARNING *** 2 or more matches to genuine L1 tracks ***";

      if (nMatch > 0) {
        tmp_matchtrk_pt = matchedTracks.at(i_track)->momentum().perp();
        tmp_matchtrk_eta = matchedTracks.at(i_track)->momentum().eta();
        tmp_matchtrk_phi = matchedTracks.at(i_track)->momentum().phi();
        tmp_matchtrk_z0 = matchedTracks.at(i_track)->z0();
        tmp_matchtrk_nFitPars = matchedTracks.at(i_track)->nFitPars();

        if (tmp_matchtrk_nFitPars == 5) {
          float tmp_matchtrk_x0 = matchedTracks.at(i_track)->POCA().x();
          float tmp_matchtrk_y0 = matchedTracks.at(i_track)->POCA().y();
          tmp_matchtrk_d0 = -tmp_matchtrk_x0 * sin(tmp_matchtrk_phi) + tmp_matchtrk_y0 * cos(tmp_matchtrk_phi);
          // tmp_matchtrk_d0 = matchedTracks.at(i_track)->d0();
        }

        tmp_matchtrk_chi2 = matchedTracks.at(i_track)->chi2();
        tmp_matchtrk_chi2dof = matchedTracks.at(i_track)->chi2Red();
        tmp_matchtrk_chi2rphi = matchedTracks.at(i_track)->chi2XYRed();
        tmp_matchtrk_chi2rz = matchedTracks.at(i_track)->chi2ZRed();
        tmp_matchtrk_bendchi2 = matchedTracks.at(i_track)->stubPtConsistency();
        tmp_matchtrk_MVA1 = matchedTracks.at(i_track)->trkMVA1();
        tmp_matchtrk_nstub = (int)matchedTracks.at(i_track)->getStubRefs().size();
        tmp_matchtrk_seed = (int)matchedTracks.at(i_track)->trackSeedType();
        tmp_matchtrk_hitpattern = (int)matchedTracks.at(i_track)->hitPattern();

        // ------------------------------------------------------------------------------------------
        tmp_matchtrk_dhits = 0;
        tmp_matchtrk_lhits = 0;

        std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
            stubRefs = matchedTracks.at(i_track)->getStubRefs();
        int tmp_nstub = stubRefs.size();

        for (int is = 0; is < tmp_nstub; is++) {
          DetId detIdStub = tGeom.idToDet((stubRefs.at(is)->clusterRef(0))->getDetId())->geographicalId();
          int layer = -999999;
          if (detIdStub.subdetId() == StripSubdetector::TOB) {
            layer = static_cast<int>(tTopo.layer(detIdStub));
            tmp_matchtrk_lhits += pow(10, layer - 1);
          } else if (detIdStub.subdetId() == StripSubdetector::TID) {
            layer = static_cast<int>(tTopo.layer(detIdStub));
            tmp_matchtrk_dhits += pow(10, layer - 1);
          }
        }
      }

      m_tp_nmatch->push_back(nMatch);

      m_matchtrk_pt->push_back(tmp_matchtrk_pt);
      m_matchtrk_eta->push_back(tmp_matchtrk_eta);
      m_matchtrk_phi->push_back(tmp_matchtrk_phi);
      m_matchtrk_z0->push_back(tmp_matchtrk_z0);
      m_matchtrk_d0->push_back(tmp_matchtrk_d0);
      m_matchtrk_chi2->push_back(tmp_matchtrk_chi2);
      m_matchtrk_chi2dof->push_back(tmp_matchtrk_chi2dof);
      m_matchtrk_chi2rphi->push_back(tmp_matchtrk_chi2rphi);
      m_matchtrk_chi2rz->push_back(tmp_matchtrk_chi2rz);
      m_matchtrk_bendchi2->push_back(tmp_matchtrk_bendchi2);
      m_matchtrk_MVA1->push_back(tmp_matchtrk_MVA1);
      m_matchtrk_nstub->push_back(tmp_matchtrk_nstub);
      m_matchtrk_dhits->push_back(tmp_matchtrk_dhits);
      m_matchtrk_lhits->push_back(tmp_matchtrk_lhits);
      m_matchtrk_seed->push_back(tmp_matchtrk_seed);
      m_matchtrk_hitpattern->push_back(tmp_matchtrk_hitpattern);
    }

    // ----------------------------------------------------------------------------------------------
    // look for L1 tracks (extended) matched to the tracking particle
    if (Displaced == "Displaced" || Displaced == "Both") {
      L1TrackPtrCollection matchedTracks = MCTruthTTTrackExtendedHandle->findTTTrackPtrs(tp_ptr);

      int nMatch = 0;
      int i_track = -1;
      float i_chi2dof = 99999;

      if (!matchedTracks.empty()) {
        if (DebugMode && (matchedTracks.size() > 1))
          edm::LogVerbatim("Tracklet") << "TrackingParticle has more than one matched L1 track!";

        // ----------------------------------------------------------------------------------------------
        // loop over matched L1 tracks
        // here, "match" means tracks that can be associated to a TrackingParticle with at least one hit of at least one of its clusters
        // https://twiki.cern.ch/twiki/bin/viewauth/CMS/SLHCTrackerTriggerSWTools#MC_truth_for_TTTrack

        for (int it = 0; it < (int)matchedTracks.size(); it++) {
          bool tmp_trk_genuine = false;
          bool tmp_trk_loosegenuine = false;
          if (MCTruthTTTrackExtendedHandle->isGenuine(matchedTracks.at(it)))
            tmp_trk_genuine = true;
          if (MCTruthTTTrackExtendedHandle->isLooselyGenuine(matchedTracks.at(it)))
            tmp_trk_loosegenuine = true;
          if (!tmp_trk_loosegenuine)
            continue;

          if (DebugMode) {
            if (MCTruthTTTrackExtendedHandle->findTrackingParticlePtr(matchedTracks.at(it)).isNull()) {
              edm::LogVerbatim("Tracklet") << "track matched to TP is NOT uniquely matched to a TP";
            } else {
              edm::Ptr<TrackingParticle> my_tp =
                  MCTruthTTTrackExtendedHandle->findTrackingParticlePtr(matchedTracks.at(it));
              edm::LogVerbatim("Tracklet") << "TP matched to track matched to TP ... tp pt = " << my_tp->p4().pt()
                                           << " eta = " << my_tp->momentum().eta()
                                           << " phi = " << my_tp->momentum().phi() << " z0 = " << my_tp->vertex().z();
            }
            edm::LogVerbatim("Tracklet") << "   ... matched L1 track has pt = "
                                         << matchedTracks.at(it)->momentum().perp()
                                         << " eta = " << matchedTracks.at(it)->momentum().eta()
                                         << " phi = " << matchedTracks.at(it)->momentum().phi()
                                         << " chi2 = " << matchedTracks.at(it)->chi2()
                                         << " consistency = " << matchedTracks.at(it)->stubPtConsistency()
                                         << " z0 = " << matchedTracks.at(it)->z0()
                                         << " nstub = " << matchedTracks.at(it)->getStubRefs().size();
            if (tmp_trk_genuine)
              edm::LogVerbatim("Tracklet") << "    (genuine!) ";
            if (tmp_trk_loosegenuine)
              edm::LogVerbatim("Tracklet") << "    (loose genuine!) ";
          }

          std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
              stubRefs = matchedTracks.at(it)->getStubRefs();
          int tmp_trk_nstub = stubRefs.size();

          if (tmp_trk_nstub < L1Tk_minNStub)
            continue;

          float dmatch_pt = 999;
          float dmatch_eta = 999;
          float dmatch_phi = 999;
          int match_id = 999;

          edm::Ptr<TrackingParticle> my_tp =
              MCTruthTTTrackExtendedHandle->findTrackingParticlePtr(matchedTracks.at(it));
          dmatch_pt = std::abs(my_tp->p4().pt() - tmp_tp_pt);
          dmatch_eta = std::abs(my_tp->p4().eta() - tmp_tp_eta);
          dmatch_phi = std::abs(my_tp->p4().phi() - tmp_tp_phi);
          match_id = my_tp->pdgId();
          float tmp_trk_chi2dof = matchedTracks.at(it)->chi2Red();

          // ensure that track is uniquely matched to the TP we are looking at!
          if (dmatch_pt < 0.1 && dmatch_eta < 0.1 && dmatch_phi < 0.1 && tmp_tp_pdgid == match_id && tmp_trk_genuine) {
            nMatch++;
            if (i_track < 0 || tmp_trk_chi2dof < i_chi2dof) {
              i_track = it;
              i_chi2dof = tmp_trk_chi2dof;
            }
          }

        }  // end loop over matched L1 tracks
      }    // end has at least 1 matched L1 track
      // ----------------------------------------------------------------------------------------------

      float tmp_matchtrkExt_pt = -999;
      float tmp_matchtrkExt_eta = -999;
      float tmp_matchtrkExt_phi = -999;
      float tmp_matchtrkExt_z0 = -999;
      float tmp_matchtrkExt_d0 = -999;
      float tmp_matchtrkExt_chi2 = -999;
      float tmp_matchtrkExt_chi2dof = -999;
      float tmp_matchtrkExt_chi2rphi = -999;
      float tmp_matchtrkExt_chi2rz = -999;
      float tmp_matchtrkExt_bendchi2 = -999;
      float tmp_matchtrkExt_MVA = -999;
      int tmp_matchtrkExt_nstub = -999;
      int tmp_matchtrkExt_dhits = -999;
      int tmp_matchtrkExt_lhits = -999;
      int tmp_matchtrkExt_seed = -999;
      int tmp_matchtrkExt_hitpattern = -999;
      int tmp_matchtrkExt_nFitPars = -999;

      if (nMatch > 1 && DebugMode)
        edm::LogVerbatim("Tracklet") << "WARNING *** 2 or more matches to genuine L1 tracks ***";

      if (nMatch > 0) {
        tmp_matchtrkExt_pt = matchedTracks.at(i_track)->momentum().perp();
        tmp_matchtrkExt_eta = matchedTracks.at(i_track)->momentum().eta();
        tmp_matchtrkExt_phi = matchedTracks.at(i_track)->momentum().phi();
        tmp_matchtrkExt_z0 = matchedTracks.at(i_track)->z0();
        tmp_matchtrkExt_nFitPars = matchedTracks.at(i_track)->nFitPars();

        if (tmp_matchtrkExt_nFitPars == 5) {
          float tmp_matchtrkExt_x0 = matchedTracks.at(i_track)->POCA().x();
          float tmp_matchtrkExt_y0 = matchedTracks.at(i_track)->POCA().y();
          tmp_matchtrkExt_d0 =
              -tmp_matchtrkExt_x0 * sin(tmp_matchtrkExt_phi) + tmp_matchtrkExt_y0 * cos(tmp_matchtrkExt_phi);
          // tmp_matchtrkExt_d0 = matchedTracks.at(i_track)->d0();
        }

        tmp_matchtrkExt_chi2 = matchedTracks.at(i_track)->chi2();
        tmp_matchtrkExt_chi2dof = matchedTracks.at(i_track)->chi2Red();
        tmp_matchtrkExt_chi2rphi = matchedTracks.at(i_track)->chi2XYRed();
        tmp_matchtrkExt_chi2rz = matchedTracks.at(i_track)->chi2ZRed();
        tmp_matchtrkExt_bendchi2 = matchedTracks.at(i_track)->stubPtConsistency();
        tmp_matchtrkExt_MVA = matchedTracks.at(i_track)->trkMVA1();
        tmp_matchtrkExt_nstub = (int)matchedTracks.at(i_track)->getStubRefs().size();
        tmp_matchtrkExt_seed = (int)matchedTracks.at(i_track)->trackSeedType();
        tmp_matchtrkExt_hitpattern = (int)matchedTracks.at(i_track)->hitPattern();

        // ------------------------------------------------------------------------------------------
        tmp_matchtrkExt_dhits = 0;
        tmp_matchtrkExt_lhits = 0;

        std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
            stubRefs = matchedTracks.at(i_track)->getStubRefs();
        int tmp_nstub = stubRefs.size();

        for (int is = 0; is < tmp_nstub; is++) {
          DetId detIdStub = tGeom.idToDet((stubRefs.at(is)->clusterRef(0))->getDetId())->geographicalId();
          int layer = -999999;
          if (detIdStub.subdetId() == StripSubdetector::TOB) {
            layer = static_cast<int>(tTopo.layer(detIdStub));
            tmp_matchtrkExt_lhits += pow(10, layer - 1);
          } else if (detIdStub.subdetId() == StripSubdetector::TID) {
            layer = static_cast<int>(tTopo.layer(detIdStub));
            tmp_matchtrkExt_dhits += pow(10, layer - 1);
          }
        }
      }

      // m_tp_nmatch->push_back(nMatch); //modify to be matches for ext
      m_matchtrkExt_pt->push_back(tmp_matchtrkExt_pt);
      m_matchtrkExt_eta->push_back(tmp_matchtrkExt_eta);
      m_matchtrkExt_phi->push_back(tmp_matchtrkExt_phi);
      m_matchtrkExt_z0->push_back(tmp_matchtrkExt_z0);
      m_matchtrkExt_d0->push_back(tmp_matchtrkExt_d0);
      m_matchtrkExt_chi2->push_back(tmp_matchtrkExt_chi2);
      m_matchtrkExt_chi2dof->push_back(tmp_matchtrkExt_chi2dof);
      m_matchtrkExt_chi2rphi->push_back(tmp_matchtrkExt_chi2rphi);
      m_matchtrkExt_chi2rz->push_back(tmp_matchtrkExt_chi2rz);
      m_matchtrkExt_bendchi2->push_back(tmp_matchtrkExt_bendchi2);
      m_matchtrkExt_MVA->push_back(tmp_matchtrkExt_MVA);
      m_matchtrkExt_nstub->push_back(tmp_matchtrkExt_nstub);
      m_matchtrkExt_dhits->push_back(tmp_matchtrkExt_dhits);
      m_matchtrkExt_lhits->push_back(tmp_matchtrkExt_lhits);
      m_matchtrkExt_seed->push_back(tmp_matchtrkExt_seed);
      m_matchtrkExt_hitpattern->push_back(tmp_matchtrkExt_hitpattern);
    }
  }  //end loop tracking particles
  trueTkMET = sqrt(trueTkMETx * trueTkMETx + trueTkMETy * trueTkMETy);

  if (L1PrimaryVertexHandle.isValid()) {
    for (vtxIter = L1PrimaryVertexHandle->begin(); vtxIter != L1PrimaryVertexHandle->end(); ++vtxIter) {
      m_pv_L1reco->push_back(vtxIter->z0());
      m_pv_L1reco_sum->push_back(vtxIter->pt());
    }
  } else
    edm::LogWarning("DataNotFound") << "\nWarning: L1PrimaryVertexHandle not found" << std::endl;

  if (L1PrimaryVertexEmuHandle.isValid()) {
    for (vtxEmuIter = L1PrimaryVertexEmuHandle->begin(); vtxEmuIter != L1PrimaryVertexEmuHandle->end(); ++vtxEmuIter) {
      m_pv_L1reco_emu->push_back(vtxEmuIter->z0());
      m_pv_L1reco_sum_emu->push_back(vtxEmuIter->pt());
    }
  } else
    edm::LogWarning("DataNotFound") << "\nWarning: L1PrimaryVertexEmuHandle not found" << std::endl;

  if (SaveTrackSums) {
    if (Displaced == "Prompt" || Displaced == "Both") {
      if (L1TkMETHandle.isValid()) {
        trkMET = L1TkMETHandle->begin()->etMiss();
        trkMETPhi = L1TkMETHandle->begin()->p4().phi();
      } else {
        edm::LogWarning("DataNotFound") << "\nWarning: tkMET handle not found" << std::endl;
      }

      if (L1TkMETEmuHandle.isValid()) {
        trkMETEmu = L1TkMETEmuHandle->begin()->hwPt() * l1tmetemu::kStepMETwordEt;
        trkMETEmuPhi = L1TkMETEmuHandle->begin()->hwPhi() * l1tmetemu::kStepMETwordPhi;
      } else {
        edm::LogWarning("DataNotFound") << "\nWarning: tkMETEmu handle not found" << std::endl;
      }

      if (L1TkMHTHandle.isValid()) {
        trkMHT = L1TkMHTHandle->begin()->EtMiss();
        trkHT = L1TkMHTHandle->begin()->etTotal();
      } else
        edm::LogWarning("DataNotFound") << "\nWarning: tkMHT handle not found" << std::endl;

      if (L1TkMHTEmuHandle.isValid()) {
        trkMHTEmu = L1TkMHTEmuHandle->begin()->p4().energy();
        trkHTEmu = L1TkMHTEmuHandle->begin()->hwPt() * l1tmhtemu::kStepMHT;
        trkMHTEmuPhi = L1TkMHTEmuHandle->begin()->hwPhi() * l1tmhtemu::kStepMHTPhi - M_PI;
      } else
        edm::LogWarning("DataNotFound") << "\nWarning: tkMHTEmu handle not found" << std::endl;
    }  //end prompt-track quantities

    if (Displaced == "Displaced" || Displaced == "Both") {
      if (L1TkMETExtendedHandle.isValid()) {
        trkMETExt = L1TkMETExtendedHandle->begin()->etMiss();
        trkMETPhiExt = L1TkMETExtendedHandle->begin()->p4().phi();
      } else {
        edm::LogWarning("DataNotFound") << "\nWarning: tkMETExtended handle not found" << std::endl;
      }

      if (L1TkMHTExtendedHandle.isValid()) {
        trkMHTExt = L1TkMHTExtendedHandle->begin()->EtMiss();
        trkHTExt = L1TkMHTExtendedHandle->begin()->etTotal();
      } else {
        edm::LogWarning("DataNotFound") << "\nWarning: tkMHTExtended handle not found" << std::endl;
      }

      if (L1TkMHTEmuExtendedHandle.isValid()) {
        trkMHTEmuExt = L1TkMHTEmuExtendedHandle->begin()->p4().energy();
        trkHTEmuExt = L1TkMHTEmuExtendedHandle->begin()->hwPt() * l1tmhtemu::kStepMHT;
        trkMHTEmuPhiExt = L1TkMHTEmuExtendedHandle->begin()->hwPhi() * l1tmhtemu::kStepMHTPhi;
      } else
        edm::LogWarning("DataNotFound") << "\nWarning: tkMHTEmuExtended handle not found" << std::endl;
    }  //end displaced-track quantities
  }

  if (SaveTrackJets) {
    if (TrackFastJetsHandle.isValid() && (Displaced == "Prompt" || Displaced == "Both")) {
      for (jetIter = TrackFastJetsHandle->begin(); jetIter != TrackFastJetsHandle->end(); ++jetIter) {
        m_trkfastjet_vz->push_back(jetIter->jetVtx());
        m_trkfastjet_ntracks->push_back(jetIter->trkPtrs().size());
        m_trkfastjet_phi->push_back(jetIter->phi());
        m_trkfastjet_eta->push_back(jetIter->eta());
        m_trkfastjet_pt->push_back(jetIter->pt());
        m_trkfastjet_p->push_back(jetIter->p());
      }
    }
    if (TrackFastJetsExtendedHandle.isValid() && (Displaced == "Displaced" || Displaced == "Both")) {
      for (jetIter = TrackFastJetsExtendedHandle->begin(); jetIter != TrackFastJetsExtendedHandle->end(); ++jetIter) {
        m_trkfastjetExt_vz->push_back(jetIter->jetVtx());
        m_trkfastjetExt_ntracks->push_back(jetIter->trkPtrs().size());
        m_trkfastjetExt_phi->push_back(jetIter->phi());
        m_trkfastjetExt_eta->push_back(jetIter->eta());
        m_trkfastjetExt_pt->push_back(jetIter->pt());
        m_trkfastjetExt_p->push_back(jetIter->p());
      }
    }
    if (!TrackJetsHandle.isValid() && (Displaced == "Prompt" || Displaced == "Both"))
      edm::LogWarning("DataNotFound") << "\nWarning: TrackJetsHandle not found" << std::endl;
    if (!TrackJetsExtendedHandle.isValid() && (Displaced == "Displaced" || Displaced == "Both"))
      edm::LogWarning("DataNotFound") << "\nWarning: TrackJetsExtendedHandle not found" << std::endl;
    if (TrackJetsHandle.isValid() && (Displaced == "Prompt" || Displaced == "Both")) {
      for (jetIter = TrackJetsHandle->begin(); jetIter != TrackJetsHandle->end(); ++jetIter) {
        m_trkjet_vz->push_back(jetIter->jetVtx());
        m_trkjet_ntracks->push_back(jetIter->ntracks());
        m_trkjet_phi->push_back(jetIter->phi());
        m_trkjet_eta->push_back(jetIter->eta());
        m_trkjet_pt->push_back(jetIter->pt());
        m_trkjet_p->push_back(jetIter->p());
        m_trkjet_nDisplaced->push_back(jetIter->nDisptracks());
        m_trkjet_nTight->push_back(jetIter->nTighttracks());
        m_trkjet_nTightDisplaced->push_back(jetIter->nTightDisptracks());
      }
    }
    if (TrackJetsExtendedHandle.isValid() && (Displaced == "Displaced" || Displaced == "Both")) {
      for (jetIter = TrackJetsExtendedHandle->begin(); jetIter != TrackJetsExtendedHandle->end(); ++jetIter) {
        m_trkjetExt_vz->push_back(jetIter->jetVtx());
        m_trkjetExt_ntracks->push_back(jetIter->ntracks());
        m_trkjetExt_phi->push_back(jetIter->phi());
        m_trkjetExt_eta->push_back(jetIter->eta());
        m_trkjetExt_pt->push_back(jetIter->pt());
        m_trkjetExt_p->push_back(jetIter->p());
        m_trkjetExt_nDisplaced->push_back(jetIter->nDisptracks());
        m_trkjetExt_nTight->push_back(jetIter->nTighttracks());
        m_trkjetExt_nTightDisplaced->push_back(jetIter->nTightDisptracks());
      }
    }

    if (!TrackJetsEmuHandle.isValid() && (Displaced == "Prompt" || Displaced == "Both"))
      edm::LogWarning("DataNotFound") << "\nWarning: TrackJetsEmuHandle not found" << std::endl;
    else if (TrackJetsEmuHandle.isValid() && (Displaced == "Prompt" || Displaced == "Both")) {
      for (jetemIter = TrackJetsEmuHandle->begin(); jetemIter != TrackJetsEmuHandle->end(); ++jetemIter) {
        m_trkjetem_ntracks->push_back(jetemIter->nt());
        m_trkjetem_phi->push_back(jetemIter->glbphi());
        m_trkjetem_eta->push_back(jetemIter->glbeta());
        m_trkjetem_pt->push_back(jetemIter->pt());
        m_trkjetem_z->push_back(jetemIter->z0());
        m_trkjetem_nxtracks->push_back(jetemIter->xt());
      }
    }
    if (!TrackJetsExtendedEmuHandle.isValid() && (Displaced == "Displaced" || Displaced == "Both"))
      edm::LogWarning("DataNotFound") << "\nWarning: TrackJetsExtendedEmuHandle not found" << std::endl;
    else if (TrackJetsExtendedEmuHandle.isValid() && (Displaced == "Displaced" || Displaced == "Both")) {
      for (jetemIter = TrackJetsExtendedEmuHandle->begin(); jetemIter != TrackJetsExtendedEmuHandle->end();
           ++jetemIter) {
        m_trkjetemExt_ntracks->push_back(jetemIter->nt());
        m_trkjetemExt_phi->push_back(jetemIter->glbphi());
        m_trkjetemExt_eta->push_back(jetemIter->glbeta());
        m_trkjetemExt_pt->push_back(jetemIter->pt());
        m_trkjetemExt_z->push_back(jetemIter->z0());
        m_trkjetemExt_nxtracks->push_back(jetemIter->xt());
      }
    }
  }  // end track jets

  eventTree->Fill();
}  // end of analyze()

int L1TrackObjectNtupleMaker::getSelectedTrackIndex(const L1TrackRef& trackRef,
                                                    const edm::Handle<L1TrackRefCollection>& selectedTrackRefs) const {
  auto it = std::find_if(selectedTrackRefs->begin(), selectedTrackRefs->end(), [&trackRef](L1TrackRef const& obj) {
    return obj == trackRef;
  });
  if (it != selectedTrackRefs->end())
    return std::distance(selectedTrackRefs->begin(), it);
  else
    return -1;
}

///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1TrackObjectNtupleMaker);
