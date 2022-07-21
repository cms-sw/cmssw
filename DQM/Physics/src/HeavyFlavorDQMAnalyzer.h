#ifndef DQMOffline_Physics_HeavyFlavorDQMAnalyzer_h
#define DQMOffline_Physics_HeavyFlavorDQMAnalyzer_h

// -*- C++ -*-
//
// Package:    DQMOffline/HeavyFlavorDQMAnalyzer
// Class:      HeavyFlavorDQMAnalyzer
//
/**\class HeavyFlavorDQMAnalyzer HeavyFlavorDQMAnalyzer.cc DQMOffline/HeavyFlavorDQMAnalyzer/plugins/HeavyFlavorDQMAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Enrico Lusiani
//         Created:  Mon, 22 Nov 2021 14:36:39 GMT
//
//

#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHTrackReference.h"

//
// class declaration
//

struct ComponentHists {
  dqm::reco::MonitorElement* h_pt;
  dqm::reco::MonitorElement* h_eta;
  dqm::reco::MonitorElement* h_phi;

  dqm::reco::MonitorElement* h_dxy;
  dqm::reco::MonitorElement* h_exy;
  dqm::reco::MonitorElement* h_dz;
  dqm::reco::MonitorElement* h_ez;

  dqm::reco::MonitorElement* h_chi2;
};

struct DecayHists {
  // kinematics
  dqm::reco::MonitorElement* h_mass;
  dqm::reco::MonitorElement* h_pt;
  dqm::reco::MonitorElement* h_eta;
  dqm::reco::MonitorElement* h_phi;

  // position
  dqm::reco::MonitorElement* h_displ2D;
  dqm::reco::MonitorElement* h_displ3D;
  dqm::reco::MonitorElement* h_sign2D;
  dqm::reco::MonitorElement* h_sign3D;

  // ct and pointing angle
  dqm::reco::MonitorElement* h_ct;
  dqm::reco::MonitorElement* h_pointing;

  // quality
  dqm::reco::MonitorElement* h_vertNormChi2;
  dqm::reco::MonitorElement* h_vertProb;

  std::vector<ComponentHists> decayComponents;
};

struct Histograms_HeavyFlavorDQMAnalyzer {
  DecayHists oniaToMuMuPrompt;
  DecayHists oniaToMuMuDispl;
  DecayHists oniaToMuMu;
  DecayHists kx0ToKPiPrompt;
  DecayHists kx0ToKPiDispl;
  DecayHists kx0ToKPi;
  DecayHists phiToKKPrompt;
  DecayHists phiToKKDispl;
  DecayHists phiToKK;
  DecayHists psi2SToJPsiPiPiPrompt;
  DecayHists psi2SToJPsiPiPiDispl;
  DecayHists psi2SToJPsiPiPi;
  DecayHists k0sToPiPi;
  DecayHists lambda0ToPPi;
  DecayHists buToJPsiK;
  DecayHists buToPsi2SK;
  DecayHists bdToJPsiKx0;
  DecayHists bsToJPsiPhi;
  DecayHists bdToJPsiK0s;
  DecayHists bcToJPsiPi;
  DecayHists lambdaBToJPsiLambda0;
};

class HeavyFlavorDQMAnalyzer : public DQMGlobalEDAnalyzer<Histograms_HeavyFlavorDQMAnalyzer> {
public:
  using Histograms = Histograms_HeavyFlavorDQMAnalyzer;

  explicit HeavyFlavorDQMAnalyzer(const edm::ParameterSet&);
  ~HeavyFlavorDQMAnalyzer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, Histograms&) const override;

  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, Histograms const&) const override;

  void bookDecayHists(DQMStore::IBooker&,
                      edm::Run const&,
                      edm::EventSetup const&,
                      DecayHists&,
                      std::string const&,
                      std::string const&,
                      int,
                      float,
                      float,
                      float distanceScaleFactor = 1.) const;
  void initComponentHists(DQMStore::IBooker&,
                          edm::Run const&,
                          edm::EventSetup const&,
                          DecayHists&,
                          TString const&) const;  // TString for the IBooker interface

  void initOniaToMuMuComponentHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, DecayHists&) const;
  void initKx0ToKPiComponentHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, DecayHists&) const;
  void initPhiToKKComponentHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, DecayHists&) const;
  void initK0sToPiPiComponentHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, DecayHists&) const;
  void initLambda0ToPPiComponentHistograms(DQMStore::IBooker&,
                                           edm::Run const&,
                                           edm::EventSetup const&,
                                           DecayHists&) const;
  void initBuToJPsiKComponentHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, DecayHists&) const;
  void initBuToPsi2SKComponentHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, DecayHists&) const;
  void initBdToJPsiKx0ComponentHistograms(DQMStore::IBooker&,
                                          edm::Run const&,
                                          edm::EventSetup const&,
                                          DecayHists&) const;
  void initBsToJPsiPhiComponentHistograms(DQMStore::IBooker&,
                                          edm::Run const&,
                                          edm::EventSetup const&,
                                          DecayHists&) const;
  void initBdToJPsiK0sComponentHistograms(DQMStore::IBooker&,
                                          edm::Run const&,
                                          edm::EventSetup const&,
                                          DecayHists&) const;
  void initBcToJPsiPiComponentHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, DecayHists&) const;
  void initLambdaBToJPsiLambda0ComponentHistograms(DQMStore::IBooker&,
                                                   edm::Run const&,
                                                   edm::EventSetup const&,
                                                   DecayHists&) const;
  void initPsi2SToJPsiPiPiComponentHistograms(DQMStore::IBooker&,
                                              edm::Run const&,
                                              edm::EventSetup const&,
                                              DecayHists&) const;

  reco::Vertex const* fillDecayHistograms(DecayHists const&,
                                          pat::CompositeCandidate const& cand,
                                          reco::VertexCollection const& pvs) const;
  void fillComponentHistograms(ComponentHists const& histos,
                               reco::Track const& component,
                               reco::BeamSpot const* bs,
                               reco::Vertex const* pv) const;

  int fillComponentHistogramsSinglePart(DecayHists const&,
                                        pat::CompositeCandidate const& cand,
                                        std::string const& name,
                                        reco::BeamSpot const* bs,
                                        reco::Vertex const* pv,
                                        int startPosition = 0) const;
  int fillComponentHistogramsLeadSoft(DecayHists const&,
                                      pat::CompositeCandidate const& cand,
                                      std::string const& name1,
                                      std::string const& name2,
                                      reco::BeamSpot const* bs,
                                      reco::Vertex const* pv,
                                      int startPosition = 0) const;

  const reco::Track* getDaughterTrack(pat::CompositeCandidate const& cand,
                                      std::string const& name,
                                      bool throwOnMissing = true) const;
  bool allTracksAvailable(pat::CompositeCandidate const& cand) const;

  int fillOniaToMuMuComponents(DecayHists const& histos,
                               pat::CompositeCandidate const& cand,
                               reco::BeamSpot const* bs,
                               reco::Vertex const* pv,
                               int startPosition = 0) const;
  int fillKx0ToKPiComponents(DecayHists const& histos,
                             pat::CompositeCandidate const& cand,
                             reco::BeamSpot const* bs,
                             reco::Vertex const* pv,
                             int startPosition = 0) const;
  int fillPhiToKKComponents(DecayHists const& histos,
                            pat::CompositeCandidate const& cand,
                            reco::BeamSpot const* bs,
                            reco::Vertex const* pv,
                            int startPosition = 0) const;
  int fillK0sToPiPiComponents(DecayHists const& histos,
                              pat::CompositeCandidate const& cand,
                              reco::BeamSpot const* bs,
                              reco::Vertex const* pv,
                              int startPosition = 0) const;
  int fillLambda0ToPPiComponents(DecayHists const& histos,
                                 pat::CompositeCandidate const& cand,
                                 reco::BeamSpot const* bs,
                                 reco::Vertex const* pv,
                                 int startPosition = 0) const;
  int fillBuToJPsiKComponents(DecayHists const& histos,
                              pat::CompositeCandidate const& cand,
                              reco::BeamSpot const* bs,
                              reco::Vertex const* pv,
                              int startPosition = 0) const;
  int fillBuToPsi2SKComponents(DecayHists const& histos,
                               pat::CompositeCandidate const& cand,
                               reco::BeamSpot const* bs,
                               reco::Vertex const* pv,
                               int startPosition = 0) const;
  int fillBdToJPsiKx0Components(DecayHists const& histos,
                                pat::CompositeCandidate const& cand,
                                reco::BeamSpot const* bs,
                                reco::Vertex const* pv,
                                int startPosition = 0) const;
  int fillBsToJPsiPhiComponents(DecayHists const& histos,
                                pat::CompositeCandidate const& cand,
                                reco::BeamSpot const* bs,
                                reco::Vertex const* pv,
                                int startPosition = 0) const;
  int fillBdToJPsiK0sComponents(DecayHists const& histos,
                                pat::CompositeCandidate const& cand,
                                reco::BeamSpot const* bs,
                                reco::Vertex const* pv,
                                int startPosition = 0) const;
  int fillBcToJPsiPiComponents(DecayHists const& histos,
                               pat::CompositeCandidate const& cand,
                               reco::BeamSpot const* bs,
                               reco::Vertex const* pv,
                               int startPosition = 0) const;
  int fillLambdaBToJPsiLambda0Components(DecayHists const& histos,
                                         pat::CompositeCandidate const& cand,
                                         reco::BeamSpot const* bs,
                                         reco::Vertex const* pv,
                                         int startPosition = 0) const;
  int fillPsi2SToJPsiPiPiComponents(DecayHists const& histos,
                                    pat::CompositeCandidate const& cand,
                                    reco::BeamSpot const* bs,
                                    reco::Vertex const* pv,
                                    int startPosition = 0) const;

  // ------------ member data ------------
  std::string folder_;

  edm::EDGetTokenT<reco::VertexCollection> pvCollectionToken;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken;

  edm::EDGetTokenT<pat::CompositeCandidateCollection> oniaToMuMuCandsToken;
  edm::EDGetTokenT<pat::CompositeCandidateCollection> kx0ToKPiCandsToken;
  edm::EDGetTokenT<pat::CompositeCandidateCollection> phiToKKCandsToken;
  edm::EDGetTokenT<pat::CompositeCandidateCollection> buToJPsiKCandsToken;
  edm::EDGetTokenT<pat::CompositeCandidateCollection> buToPsi2SKCandsToken;
  edm::EDGetTokenT<pat::CompositeCandidateCollection> bdToJPsiKx0CandsToken;
  edm::EDGetTokenT<pat::CompositeCandidateCollection> bsToJPsiPhiCandsToken;
  edm::EDGetTokenT<pat::CompositeCandidateCollection> k0sToPiPiCandsToken;
  edm::EDGetTokenT<pat::CompositeCandidateCollection> lambda0ToPPiCandsToken;
  edm::EDGetTokenT<pat::CompositeCandidateCollection> bdToJPsiK0sCandsToken;
  edm::EDGetTokenT<pat::CompositeCandidateCollection> lambdaBToJPsiLambda0CandsToken;
  edm::EDGetTokenT<pat::CompositeCandidateCollection> bcToJPsiPiCandsToken;
  edm::EDGetTokenT<pat::CompositeCandidateCollection> psi2SToJPsiPiPiCandsToken;
};

#endif
