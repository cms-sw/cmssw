// -*- C++ -*-
//
// Package:    Alignment/OfflineValidation
// Class:      DiElectronVertexValidation
//
/**\class DiElectronVertexValidation DiElectronVertexValidation.cc Alignment/OfflineValidation/plugins/DiElectronVertexValidation.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Musich
//         Created:  Thu, 13 May 2021 10:24:07 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

// electrons
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

// tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

// vertices
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"

// TFileService
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// utilities
#include "DataFormats/Math/interface/deltaR.h"
#include "Alignment/OfflineValidation/interface/DiLeptonVertexHelpers.h"

// ROOT
#include "TLorentzVector.h"
#include "TH1F.h"
#include "TH2F.h"

//
// class declaration
//
class DiElectronVertexValidation : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit DiElectronVertexValidation(const edm::ParameterSet&);
  ~DiElectronVertexValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  bool passLooseSelection(const reco::GsfElectron& electron);

  // ----------member data ---------------------------
  DiLeptonHelp::Counts myCounts;
  std::vector<double> pTthresholds_;
  float maxSVdist_;

  // plot configurations

  edm::ParameterSet CosPhiConfiguration_;
  edm::ParameterSet CosPhi3DConfiguration_;
  edm::ParameterSet VtxProbConfiguration_;
  edm::ParameterSet VtxDistConfiguration_;
  edm::ParameterSet VtxDist3DConfiguration_;
  edm::ParameterSet VtxDistSigConfiguration_;
  edm::ParameterSet VtxDist3DSigConfiguration_;
  edm::ParameterSet DiMuMassConfiguration_;

  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbESToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection>
      electronsToken_;  //used to select what electrons to read from configuration
  edm::EDGetTokenT<reco::GsfTrackCollection>
      gsfTracksToken_;                                    //used to select what tracks to read from configuration file
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;  //used to select what vertices to read from configuration file

  TH1F* hSVProb_;
  TH1F* hSVDist_;
  TH1F* hGSFMult_;
  TH1F* hGSFMultAftPt_;
  TH1F* hGSF0Pt_;
  TH1F* hGSF0Eta_;
  TH1F* hGSF1Pt_;
  TH1F* hGSF1Eta_;
  TH1F* hSVDistSig_;
  TH1F* hSVDist3D_;
  TH1F* hSVDist3DSig_;
  TH1F* hCosPhi_;
  TH1F* hCosPhi3D_;
  TH1F* hTrackInvMass_;
  TH1F* hInvMass_;
  TH1I* hClosestVtxIndex_;
  TH1F* hCutFlow_;

  // 2D maps

  DiLeptonHelp::PlotsVsKinematics CosPhiPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::EE);
  DiLeptonHelp::PlotsVsKinematics CosPhi3DPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::EE);
  DiLeptonHelp::PlotsVsKinematics VtxProbPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::EE);
  DiLeptonHelp::PlotsVsKinematics VtxDistPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::EE);
  DiLeptonHelp::PlotsVsKinematics VtxDist3DPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::EE);
  DiLeptonHelp::PlotsVsKinematics VtxDistSigPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::EE);
  DiLeptonHelp::PlotsVsKinematics VtxDist3DSigPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::EE);
  DiLeptonHelp::PlotsVsKinematics ZMassPlots = DiLeptonHelp::PlotsVsKinematics(DiLeptonHelp::EE);
};

//
// constants, enums and typedefs
//

static constexpr float cmToum = 10e4;
static constexpr float emass2 = 0.0005109990615 * 0.0005109990615;  //electron mass squared  (GeV^2/c^4)

//
// constructors and destructor
//
DiElectronVertexValidation::DiElectronVertexValidation(const edm::ParameterSet& iConfig)
    : pTthresholds_(iConfig.getParameter<std::vector<double>>("pTThresholds")),
      maxSVdist_(iConfig.getParameter<double>("maxSVdist")),
      CosPhiConfiguration_(iConfig.getParameter<edm::ParameterSet>("CosPhiConfig")),
      CosPhi3DConfiguration_(iConfig.getParameter<edm::ParameterSet>("CosPhi3DConfig")),
      VtxProbConfiguration_(iConfig.getParameter<edm::ParameterSet>("VtxProbConfig")),
      VtxDistConfiguration_(iConfig.getParameter<edm::ParameterSet>("VtxDistConfig")),
      VtxDist3DConfiguration_(iConfig.getParameter<edm::ParameterSet>("VtxDist3DConfig")),
      VtxDistSigConfiguration_(iConfig.getParameter<edm::ParameterSet>("VtxDistSigConfig")),
      VtxDist3DSigConfiguration_(iConfig.getParameter<edm::ParameterSet>("VtxDist3DSigConfig")),
      DiMuMassConfiguration_(iConfig.getParameter<edm::ParameterSet>("DiMuMassConfig")),
      ttbESToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      electronsToken_(consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons"))),
      gsfTracksToken_(consumes<reco::GsfTrackCollection>(iConfig.getParameter<edm::InputTag>("gsfTracks"))),
      vertexToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))) {
  usesResource(TFileService::kSharedResource);

  // sort the vector of thresholds
  std::sort(pTthresholds_.begin(), pTthresholds_.end(), [](const double& lhs, const double& rhs) { return lhs > rhs; });

  edm::LogInfo("DiElectronVertexValidation") << __FUNCTION__;
  for (const auto& thr : pTthresholds_) {
    edm::LogInfo("DiElectronVertexValidation") << " Threshold: " << thr << " ";
  }
  edm::LogInfo("DiElectronVertexValidation") << "Max SV distance: " << maxSVdist_ << " ";
}

DiElectronVertexValidation::~DiElectronVertexValidation() = default;

//
// member functions
//

// ------------ method called for each event  ------------
void DiElectronVertexValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  myCounts.eventsTotal++;

  std::vector<const reco::GsfElectron*> myGoodGsfElectrons;

  int totGsfCounter = 0;
  for (const auto& gsfEle : iEvent.get(electronsToken_)) {
    totGsfCounter++;
    if (gsfEle.pt() > pTthresholds_[1] && passLooseSelection(gsfEle)) {
      myGoodGsfElectrons.emplace_back(&gsfEle);
    }
  }

  hGSFMult_->Fill(totGsfCounter);

  std::sort(myGoodGsfElectrons.begin(),
            myGoodGsfElectrons.end(),
            [](const reco::GsfElectron*& lhs, const reco::GsfElectron*& rhs) { return lhs->pt() > rhs->pt(); });

  hGSFMultAftPt_->Fill(myGoodGsfElectrons.size());

  // reject if there's no Z
  if (myGoodGsfElectrons.size() < 2)
    return;

  myCounts.eventsAfterMult++;

  if ((myGoodGsfElectrons[0]->pt()) < pTthresholds_[0] || (myGoodGsfElectrons[1]->pt() < pTthresholds_[1]))
    return;

  myCounts.eventsAfterPt++;

  if (myGoodGsfElectrons[0]->charge() * myGoodGsfElectrons[1]->charge() > 0)
    return;

  if (std::max(std::abs(myGoodGsfElectrons[0]->eta()), std::abs(myGoodGsfElectrons[1]->eta())) > 2.4)
    return;

  myCounts.eventsAfterEta++;

  const auto& ele1 = myGoodGsfElectrons[1]->p4();
  const auto& ele0 = myGoodGsfElectrons[0]->p4();
  const auto& mother = ele1 + ele0;

  float invMass = mother.M();
  hInvMass_->Fill(invMass);

  // just copy the top two muons
  std::vector<const reco::GsfElectron*> theZElectronVector;
  theZElectronVector.reserve(2);
  theZElectronVector.emplace_back(myGoodGsfElectrons[1]);
  theZElectronVector.emplace_back(myGoodGsfElectrons[0]);

  // do the matching of Z muons with inner tracks

  std::vector<const reco::GsfTrack*> myGoodGsfTracks;

  for (const auto& electron : theZElectronVector) {
    float minD = 1000.;
    const reco::GsfTrack* theMatch = nullptr;
    for (const auto& track : iEvent.get(gsfTracksToken_)) {
      float D = ::deltaR(electron->gsfTrack()->eta(), electron->gsfTrack()->phi(), track.eta(), track.phi());
      if (D < minD) {
        minD = D;
        theMatch = &track;
      }
    }
    myGoodGsfTracks.emplace_back(theMatch);
  }

  hGSF0Pt_->Fill(myGoodGsfTracks[0]->pt());
  hGSF0Eta_->Fill(myGoodGsfTracks[0]->eta());
  hGSF1Pt_->Fill(myGoodGsfTracks[1]->pt());
  hGSF1Eta_->Fill(myGoodGsfTracks[1]->eta());

  const TransientTrackBuilder* theB = &iSetup.getData(ttbESToken_);
  TransientVertex aTransVtx;
  std::vector<reco::TransientTrack> tks;

  std::vector<const reco::GsfTrack*> myTracks;
  myTracks.emplace_back(myGoodGsfTracks[0]);
  myTracks.emplace_back(myGoodGsfTracks[1]);

  if (myTracks.size() != 2)
    return;

  const auto& e1 = myTracks[1]->momentum();
  const auto& e0 = myTracks[0]->momentum();
  const auto& ditrack = e1 + e0;

  const auto& tplus = myTracks[0]->charge() > 0 ? myTracks[0] : myTracks[1];
  const auto& tminus = myTracks[0]->charge() < 0 ? myTracks[0] : myTracks[1];

  TLorentzVector p4_tplus(tplus->px(), tplus->py(), tplus->pz(), sqrt((tplus->p() * tplus->p()) + emass2));
  TLorentzVector p4_tminus(tminus->px(), tminus->py(), tminus->pz(), sqrt((tminus->p() * tminus->p()) + emass2));

  // creat the pair of TLorentVectors used to make the plos
  std::pair<TLorentzVector, TLorentzVector> tktk_p4 = std::make_pair(p4_tplus, p4_tminus);

  const auto& Zp4 = p4_tplus + p4_tminus;
  float track_invMass = Zp4.M();
  hTrackInvMass_->Fill(track_invMass);

  // fill the z->mm mass plots
  ZMassPlots.fillPlots(track_invMass, tktk_p4);

  math::XYZPoint ZpT(ditrack.x(), ditrack.y(), 0);
  math::XYZPoint Zp(ditrack.x(), ditrack.y(), ditrack.z());

  for (const auto& track : myTracks) {
    reco::TransientTrack trajectory = theB->build(track);
    tks.push_back(trajectory);
  }

  KalmanVertexFitter kalman(true);
  aTransVtx = kalman.vertex(tks);

  double SVProb = TMath::Prob(aTransVtx.totalChiSquared(), (int)aTransVtx.degreesOfFreedom());
  hSVProb_->Fill(SVProb);

  if (!aTransVtx.isValid())
    return;

  myCounts.eventsAfterVtx++;

  // fill the VtxProb plots
  VtxProbPlots.fillPlots(SVProb, tktk_p4);

  // get collection of reconstructed vertices from event
  edm::Handle<reco::VertexCollection> vertexHandle = iEvent.getHandle(vertexToken_);

  math::XYZPoint mainVtx(0, 0, 0);
  reco::Vertex TheMainVtx;  // = vertexHandle.product()->front();

  VertexDistanceXY vertTool;
  VertexDistance3D vertTool3D;

  if (vertexHandle.isValid()) {
    const reco::VertexCollection* vertices = vertexHandle.product();
    float minD = 9999.;
    int closestVtxIndex = 0;
    int counter = 0;
    for (const auto& vtx : *vertices) {
      double dist3D = vertTool3D.distance(aTransVtx, vtx).value();
      if (dist3D < minD) {
        minD = dist3D;
        closestVtxIndex = counter;
      }
      counter++;
    }
    if ((*vertices).at(closestVtxIndex).isValid()) {
      hClosestVtxIndex_->Fill(closestVtxIndex);
      TheMainVtx = (*vertices).at(closestVtxIndex);
      mainVtx.SetXYZ(TheMainVtx.position().x(), TheMainVtx.position().y(), TheMainVtx.position().z());
    }
  }

  const math::XYZPoint myVertex(aTransVtx.position().x(), aTransVtx.position().y(), aTransVtx.position().z());
  const math::XYZPoint deltaVtx(mainVtx.x() - myVertex.x(), mainVtx.y() - myVertex.y(), mainVtx.z() - myVertex.z());

  if (TheMainVtx.isValid()) {
    // Z Vertex distance in the xy plane
    double distance = vertTool.distance(aTransVtx, TheMainVtx).value();
    double dist_err = vertTool.distance(aTransVtx, TheMainVtx).error();

    hSVDist_->Fill(distance * cmToum);
    hSVDistSig_->Fill(distance / dist_err);

    // fill the VtxDist plots
    VtxDistPlots.fillPlots(distance * cmToum, tktk_p4);

    // fill the VtxDisSig plots
    VtxDistSigPlots.fillPlots(distance / dist_err, tktk_p4);

    // Z Vertex distance in 3D
    double distance3D = vertTool3D.distance(aTransVtx, TheMainVtx).value();
    double dist3D_err = vertTool3D.distance(aTransVtx, TheMainVtx).error();

    hSVDist3D_->Fill(distance3D * cmToum);
    hSVDist3DSig_->Fill(distance3D / dist3D_err);

    // fill the VtxDist3D plots
    VtxDist3DPlots.fillPlots(distance3D * cmToum, tktk_p4);

    // fill the VtxDisSig plots
    VtxDist3DSigPlots.fillPlots(distance3D / dist3D_err, tktk_p4);

    // cut on the PV - SV distance
    if (distance * cmToum < maxSVdist_) {
      myCounts.eventsAfterDist++;

      double cosphi = (ZpT.x() * deltaVtx.x() + ZpT.y() * deltaVtx.y()) /
                      (sqrt(ZpT.x() * ZpT.x() + ZpT.y() * ZpT.y()) *
                       sqrt(deltaVtx.x() * deltaVtx.x() + deltaVtx.y() * deltaVtx.y()));

      double cosphi3D = (Zp.x() * deltaVtx.x() + Zp.y() * deltaVtx.y() + Zp.z() * deltaVtx.z()) /
                        (sqrt(Zp.x() * Zp.x() + Zp.y() * Zp.y() + Zp.z() * Zp.z()) *
                         sqrt(deltaVtx.x() * deltaVtx.x() + deltaVtx.y() * deltaVtx.y() + deltaVtx.z() * deltaVtx.z()));

      hCosPhi_->Fill(cosphi);
      hCosPhi3D_->Fill(cosphi3D);

      // fill the cosphi plots
      CosPhiPlots.fillPlots(cosphi, tktk_p4);

      // fill the VtxDisSig plots
      CosPhi3DPlots.fillPlots(cosphi3D, tktk_p4);
    }
  }
}

bool DiElectronVertexValidation::passLooseSelection(const reco::GsfElectron& el) {
  float dEtaln = fabs(el.deltaEtaSuperClusterTrackAtVtx());
  float dPhiln = fabs(el.deltaPhiSuperClusterTrackAtVtx());
  float sigmaletaleta = el.full5x5_sigmaIetaIeta();
  float hem = el.hadronicOverEm();
  double resol = fabs((1 / el.ecalEnergy()) - (el.eSuperClusterOverP() / el.ecalEnergy()));
  double mHits = el.gsfTrack()->hitPattern().numberOfAllHits(reco::HitPattern::MISSING_INNER_HITS);
  bool barrel = (fabs(el.superCluster()->eta()) <= 1.479);
  bool endcap = (!barrel && fabs(el.superCluster()->eta()) < 2.5);

  // loose electron ID

  if (barrel && dEtaln < 0.00477 && dPhiln < 0.222 && sigmaletaleta < 0.011 && hem < 0.298 && resol < 0.241 &&
      mHits <= 1)
    return true;
  if (endcap && dEtaln < 0.00868 && dPhiln < 0.213 && sigmaletaleta < 0.0314 && hem < 0.101 && resol < 0.14 &&
      mHits <= 1)
    return true;

  return false;
}

// ------------ method called once each job just before starting event loop  ------------
void DiElectronVertexValidation::beginJob() {
  // please remove this method if not needed
  edm::Service<TFileService> fs;

  // clang-format off
  TH1F::SetDefaultSumw2(kTRUE);
  
  hGSFMult_= fs->make<TH1F>("GSFMult", ";# gsf tracks;N. events", 20, 0., 20.);
  hGSFMultAftPt_= fs->make<TH1F>("GSFMultAftPt", ";# gsf tracks;N. events", 20, 0., 20.);
  hGSF0Pt_=  fs->make<TH1F>("GSF0Pt", ";leading GSF track p_{T};N. GSF tracks", 100, 0., 100.);
  hGSF0Eta_= fs->make<TH1F>("GSF0Eta", ";leading GSF track #eta;N. GSF tracks", 50, -2.5, 2.5);
  hGSF1Pt_=  fs->make<TH1F>("GSF1Pt", ";sub-leading GSF track p_{T};N. GSF tracks", 100, 0., 100.);
  hGSF1Eta_= fs->make<TH1F>("GSF1Eta", ";sub-leading GSF track #eta;N. GSF tracks", 50, -2.5, 2.5);

  hSVProb_ = fs->make<TH1F>("VtxProb", ";ZV vertex probability;N(e^{+}e^{-} pairs)", 100, 0., 1.);

  hSVDist_ = fs->make<TH1F>("VtxDist", ";PV-ZV xy distance [#mum];N(e^{+}e^{-} pairs)", 100, 0., 1000.);
  hSVDistSig_ = fs->make<TH1F>("VtxDistSig", ";PV-ZV xy distance signficance;N(e^{+}e^{-} pairs)", 100, 0., 5.);

  hSVDist3D_ = fs->make<TH1F>("VtxDist3D", ";PV-ZV 3D distance [#mum];N(e^{+}e^{-} pairs)", 100, 0., 1000.);
  hSVDist3DSig_ = fs->make<TH1F>("VtxDist3DSig", ";PV-ZV 3D distance signficance;N(e^{+}e^{-} pairs)", 100, 0., 5.);

  hCosPhi_ = fs->make<TH1F>("CosPhi", ";cos(#phi_{xy});N(ee pairs)", 50, -1., 1.);
  hCosPhi3D_ = fs->make<TH1F>("CosPhi3D", ";cos(#phi_{3D});N(ee pairs)", 50, -1., 1.);
  hTrackInvMass_ = fs->make<TH1F>("TkTkInvMass", ";M(tk,tk) [GeV];N(tk tk pairs)", 70., 50., 120.);
  hInvMass_ = fs->make<TH1F>("InvMass", ";M(#mu#mu) [GeV];N(#mu#mu pairs)", 70., 50., 120.);

  hClosestVtxIndex_ = fs->make<TH1I>("ClosestVtxIndex", ";closest vertex index;N(tk tk pairs)", 20, -0.5, 19.5);

  // 2D Maps

  TFileDirectory dirCosPhi = fs->mkdir("CosPhiPlots");
  CosPhiPlots.bookFromPSet(dirCosPhi, CosPhiConfiguration_);

  TFileDirectory dirCosPhi3D = fs->mkdir("CosPhi3DPlots");
  CosPhi3DPlots.bookFromPSet(dirCosPhi3D, CosPhi3DConfiguration_);

  TFileDirectory dirVtxProb = fs->mkdir("VtxProbPlots");
  VtxProbPlots.bookFromPSet(dirVtxProb, VtxProbConfiguration_);

  TFileDirectory dirVtxDist = fs->mkdir("VtxDistPlots");
  VtxDistPlots.bookFromPSet(dirVtxDist, VtxDistConfiguration_);

  TFileDirectory dirVtxDist3D = fs->mkdir("VtxDist3DPlots");
  VtxDist3DPlots.bookFromPSet(dirVtxDist3D, VtxDist3DConfiguration_);

  TFileDirectory dirVtxDistSig = fs->mkdir("VtxDistSigPlots");
  VtxDistSigPlots.bookFromPSet(dirVtxDistSig, VtxDistSigConfiguration_);

  TFileDirectory dirVtxDist3DSig = fs->mkdir("VtxDist3DSigPlots");
  VtxDist3DSigPlots.bookFromPSet(dirVtxDist3DSig, VtxDist3DSigConfiguration_);

  TFileDirectory dirInvariantMass = fs->mkdir("InvariantMassPlots");
  ZMassPlots.bookFromPSet(dirInvariantMass, DiMuMassConfiguration_);

  // cut flow 

  hCutFlow_ = fs->make<TH1F>("hCutFlow","cut flow;cut step;events left",6,-0.5,5.5);
  std::string names[6]={"Total","Mult.",">pT","<eta","hasVtx","VtxDist"};
  for(unsigned int i=0;i<6;i++){
    hCutFlow_->GetXaxis()->SetBinLabel(i+1,names[i].c_str());
  }

  myCounts.zeroAll();
}

// ------------ method called once each job just after ending the event loop  ------------
void DiElectronVertexValidation::endJob() {
  myCounts.printCounts();

  hCutFlow_->SetBinContent(1,myCounts.eventsTotal);
  hCutFlow_->SetBinContent(2,myCounts.eventsAfterMult);
  hCutFlow_->SetBinContent(3,myCounts.eventsAfterPt);
  hCutFlow_->SetBinContent(4,myCounts.eventsAfterEta);
  hCutFlow_->SetBinContent(5,myCounts.eventsAfterVtx);
  hCutFlow_->SetBinContent(6,myCounts.eventsAfterDist);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DiElectronVertexValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gsfTracks",edm::InputTag("electronGsfTracks"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("gedGsfElectrons"));
  desc.add<std::vector<double>>("pTThresholds", {25., 15.});
  desc.add<double>("maxSVdist", 50.);

  {
    edm::ParameterSetDescription psDiMuMass;
    psDiMuMass.add<std::string>("name", "DiMuMass");
    psDiMuMass.add<std::string>("title", "M(#mu#mu)");
    psDiMuMass.add<std::string>("yUnits", "[GeV]");
    psDiMuMass.add<int>("NxBins", 24);
    psDiMuMass.add<int>("NyBins", 50);
    psDiMuMass.add<double>("ymin", 70.);
    psDiMuMass.add<double>("ymax", 120.);
    desc.add<edm::ParameterSetDescription>("DiMuMassConfig", psDiMuMass);
  }
  {
    edm::ParameterSetDescription psCosPhi;
    psCosPhi.add<std::string>("name", "CosPhi");
    psCosPhi.add<std::string>("title", "cos(#phi_{xy})");
    psCosPhi.add<std::string>("yUnits", "");
    psCosPhi.add<int>("NxBins", 50);
    psCosPhi.add<int>("NyBins", 50);
    psCosPhi.add<double>("ymin", -1.);
    psCosPhi.add<double>("ymax", 1.);
    desc.add<edm::ParameterSetDescription>("CosPhiConfig", psCosPhi);
  }
  {
    edm::ParameterSetDescription psCosPhi3D;
    psCosPhi3D.add<std::string>("name", "CosPhi3D");
    psCosPhi3D.add<std::string>("title", "cos(#phi_{3D})");
    psCosPhi3D.add<std::string>("yUnits", "");
    psCosPhi3D.add<int>("NxBins", 50);
    psCosPhi3D.add<int>("NyBins", 50);
    psCosPhi3D.add<double>("ymin", -1.);
    psCosPhi3D.add<double>("ymax", 1.);
    desc.add<edm::ParameterSetDescription>("CosPhi3DConfig", psCosPhi3D);
  }
  {
    edm::ParameterSetDescription psVtxProb;
    psVtxProb.add<std::string>("name", "VtxProb");
    psVtxProb.add<std::string>("title", "Prob(#chi^{2}_{SV})");
    psVtxProb.add<std::string>("yUnits", "");
    psVtxProb.add<int>("NxBins", 50);
    psVtxProb.add<int>("NyBins", 50);
    psVtxProb.add<double>("ymin", 0);
    psVtxProb.add<double>("ymax", 1.);
    desc.add<edm::ParameterSetDescription>("VtxProbConfig", psVtxProb);
  }
  {
    edm::ParameterSetDescription psVtxDist;
    psVtxDist.add<std::string>("name", "VtxDist");
    psVtxDist.add<std::string>("title", "d_{xy}(PV,SV)");
    psVtxDist.add<std::string>("yUnits", "[#mum]");
    psVtxDist.add<int>("NxBins", 50);
    psVtxDist.add<int>("NyBins", 100);
    psVtxDist.add<double>("ymin", 0);
    psVtxDist.add<double>("ymax", 300.);
    desc.add<edm::ParameterSetDescription>("VtxDistConfig", psVtxDist);
  }
  {
    edm::ParameterSetDescription psVtxDist3D;
    psVtxDist3D.add<std::string>("name", "VtxDist3D");
    psVtxDist3D.add<std::string>("title", "d_{3D}(PV,SV)");
    psVtxDist3D.add<std::string>("yUnits", "[#mum]");
    psVtxDist3D.add<int>("NxBins", 50);
    psVtxDist3D.add<int>("NyBins", 250);
    psVtxDist3D.add<double>("ymin", 0);
    psVtxDist3D.add<double>("ymax", 500.);
    desc.add<edm::ParameterSetDescription>("VtxDist3DConfig", psVtxDist3D);
  }
  {
    edm::ParameterSetDescription psVtxDistSig;
    psVtxDistSig.add<std::string>("name", "VtxDistSig");
    psVtxDistSig.add<std::string>("title", "d_{xy}(PV,SV)/#sigma_{dxy}(PV,SV)");
    psVtxDistSig.add<std::string>("yUnits", "");
    psVtxDistSig.add<int>("NxBins", 50);
    psVtxDistSig.add<int>("NyBins", 100);
    psVtxDistSig.add<double>("ymin", 0);
    psVtxDistSig.add<double>("ymax", 5.);
    desc.add<edm::ParameterSetDescription>("VtxDistSigConfig", psVtxDistSig);
  }
  {
    edm::ParameterSetDescription psVtxDist3DSig;
    psVtxDist3DSig.add<std::string>("name", "VtxDist3DSig");
    psVtxDist3DSig.add<std::string>("title", "d_{3D}(PV,SV)/#sigma_{d3D}(PV,SV)");
    psVtxDist3DSig.add<std::string>("yUnits", "");
    psVtxDist3DSig.add<int>("NxBins", 50);
    psVtxDist3DSig.add<int>("NyBins", 100);
    psVtxDist3DSig.add<double>("ymin", 0);
    psVtxDist3DSig.add<double>("ymax", 5.);
    desc.add<edm::ParameterSetDescription>("VtxDist3DSigConfig", psVtxDist3DSig);
  }

  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DiElectronVertexValidation);
