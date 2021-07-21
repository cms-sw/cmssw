// -*- C++ -*-
//
// Package:    SkimmingForB/LeptonSkimming
// Class:      LeptonSkimming
//
/**\class LeptonSkimming LeptonSkimming.cc SkimmingForB/LeptonSkimming/plugins/LeptonSkimming.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Georgios Karathanasis georgios.karathanasis@cern.ch
//         Created:  Thu, 29 Nov 2018 15:23:09 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "CommonTools/Egamma/interface/ConversionTools.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateIsolation.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/ParametrizedEngine/src/OAEParametrizedMagneticField.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "L1Trigger/L1TNtuples/interface/MuonID.h"
#include <vector>
#include <string>
#include <iostream>
#include "DataFormats/Common/interface/Ref.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "TLorentzVector.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <cmath>

//
// class declaration
//

class LeptonSkimming : public edm::stream::EDFilter<> {
public:
  explicit LeptonSkimming(const edm::ParameterSet&);
  ~LeptonSkimming() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  bool hltFired(const edm::Event& iEvent, const edm::EventSetup& iSetup, std::vector<string> HLTPath);
  std::array<float, 5> hltObject(const edm::Event& iEvent, const edm::EventSetup& iSetup, std::vector<string> Seed);

  edm::EDGetToken electronsToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> eleBWPToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> eleUnBWPToken_;
  edm::EDGetToken muonsToken_;
  edm::EDGetToken Tracks_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<reco::ConversionCollection> conversionsToken_;
  edm::EDGetTokenT<edm::TriggerResults> trgresultsToken_;
  edm::EDGetTokenT<trigger::TriggerEvent> trigobjectsToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bFieldToken_;
  std::vector<string> HLTFilter_, HLTPath_;

  bool Result = false;
  float beam_x, beam_y, beam_z;

  std::vector<float> track_vx, track_vy, track_vz;
  std::vector<float> muon_pt, muon_eta, muon_phi, el_pt, el_eta, el_phi, muon_soft, muon_medium, muon_tight;
  std::vector<unsigned int> Epair_ObjectIndex, Epair_TrkIndex, Epair_ObjectId;

  std::vector<float> SelectedTrgObj_PtEtaPhiCharge;
  int SelectedMu_index;

  float SelectedMu_DR = -1;
  float ZvertexTrg = -1 * std::numeric_limits<float>::max();
  unsigned int trk_index = 0;
  reco::TrackBase::Point vertex_point;
  std::vector<float> tempPtEtaPhiM, tempXYZ;
  std::vector<std::shared_ptr<reco::Track>> cleanedObjTracks;
  std::vector<std::shared_ptr<reco::Track>> cleanedPairTracks;
  std::vector<std::shared_ptr<reco::Track>> MuTracks;
  std::vector<std::shared_ptr<reco::Track>> ElTracks;
  std::vector<std::shared_ptr<reco::Track>> cleanedTracks;
  std::vector<unsigned int> trackObj_container, trackPair_container;
  std::vector<unsigned int> Trk_container, object_container, object_id;

  unsigned int nel = 0, nmuons = 0, ntracks = 0;
  //options
  double PtTrack_Cut = 0;
  double EtaTrack_Cut = 10;
  double MinChi2Track_Cut = -1000;
  double MaxChi2Track_Cut = 1000;
  double MuTrkMinDR_Cut = 0;
  double MaxMee_Cut = 1000;
  double PtKTrack_Cut = 0;
  double MaxMB_Cut = 1000;
  double MinMB_Cut = 0;
  bool SaveOnlyTracks = false;
  bool SaveOnlyEPairTracks = false;
  double TrackSdxy_Cut = 0;
  bool UseOnlyBKeeMCForTriplets = false;
  double MinMee_Cut = 0;
  double Probee_Cut = 0;
  double Cosee_Cut = -1;
  bool EarlyStop = false;
  double MuTrgMatchCone = 1000;
  bool SkipIfNoMuMatch = false;
  double EpairZvtx_Cut = 10000000;
  double Ksdxy_Cut = 1000;
  double ProbeeK_Cut = 0;
  double CoseeK_Cut = 0;
  double TrackMuDz_Cut = 100000000;
  double TrgExclusionCone = -1;
  double SLxy_Cut = 0;
  double PtB_Cut = 0;
  double PtMu_Cut = 0;
  double QualMu_Cut = 0;
  double PtEl_Cut = 0;
  double MuTrgExclusionCone = 0;
  double ElTrgExclusionCone = 0;
  double TrkObjExclusionCone = 0;
  double MuTrgMuDz_Cut = 1000;
  double ElTrgMuDz_Cut = 1000;
  bool SaveOutputRoot = true;
  bool ObjPtLargerThanTrack = false;
  double BiasedWP = -100;
  double UnbiasedWP = -100;
  bool SkimOnlyMuons = false;
  bool SkimOnlyElectrons = false;
  int test_ev = 0;
  // ----------member data ---------------------------
};
