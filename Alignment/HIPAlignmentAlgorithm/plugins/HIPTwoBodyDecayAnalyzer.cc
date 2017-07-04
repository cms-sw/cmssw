#include <iostream>
#include <vector>
#include <memory>
#include <utility>
#include <cmath>
#include <string>
#include "TString.h"
#include "TLorentzVector.h"
#include "TTree.h"
#include "TH1F.h"
#include "TMath.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"


class HIPTwoBodyDecayAnalyzer : public edm::EDAnalyzer {
public:
  
  explicit HIPTwoBodyDecayAnalyzer(const edm::ParameterSet&);
  ~HIPTwoBodyDecayAnalyzer();

  edm::EDGetTokenT<reco::TrackCollection> alcareco_trackCollToken_;
  edm::EDGetTokenT<reco::TrackCollection> refit1_trackCollToken_;
  edm::EDGetTokenT<reco::TrackCollection> ctf_trackCollToken_;
  edm::EDGetTokenT<reco::TrackCollection> final_trackCollToken_;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  TTree* tree;
  std::vector<std::pair<TString, Float_t*>> floatBranches;
  std::vector<std::pair<TString, Int_t*>> intBranches;
  std::vector<std::pair<TString, Short_t*>> shortBranches;

private:

  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;

  void bookAllBranches();
  bool bookBranch(TString bname, TString btype);
  TString searchArray(TString branchname, int& position);
  template<typename varType> void setVal(TString bname, varType value){
    int varposition=-1;
    TString varbranchtype = searchArray(bname, varposition);
    if (varposition==-1) std::cerr << "HIPTwoBodyDecayAnalyzer::setVal -> Could not find the branch called " << bname << "!" << std::endl;
    else if (varbranchtype=="s") *(shortBranches.at(varposition).second)=value;
    else if (varbranchtype=="i") *(intBranches.at(varposition).second)=value;
    else if (varbranchtype=="f") *(floatBranches.at(varposition).second)=value;
    else std::cerr << "HIPTwoBodyDecayAnalyzer::setVal -> Could not find the type " << varbranchtype << " for branch called " << bname << "!" << std::endl;
  }
  void cleanBranches();
  void initializeBranches();
  bool actuateBranches();

  void analyzeTrackCollection(
    TString strTrackType,
    edm::ESHandle<TransientTrackBuilder>& theTTBuilder,
    edm::Handle<reco::TrackCollection>& hTrackColl,
    bool verbose=false
    );
  reco::Vertex fitDimuonVertex(
    edm::ESHandle<TransientTrackBuilder>& theTTBuilder,
    edm::Handle<reco::TrackCollection>& hTrackColl,
    bool& fitOk
    );

};

HIPTwoBodyDecayAnalyzer::HIPTwoBodyDecayAnalyzer(const edm::ParameterSet& iConfig){
  alcareco_trackCollToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("alcarecotracks"));
  refit1_trackCollToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("refit1tracks"));
  ctf_trackCollToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("refit2tracks"));
  final_trackCollToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("finaltracks"));

  edm::Service<TFileService> fs;

  tree = fs->make<TTree>("TestTree", "");
  bookAllBranches();
}


TString HIPTwoBodyDecayAnalyzer::searchArray(TString branchname, int& position){
  for (unsigned short el=0; el<shortBranches.size(); el++){
    if (branchname==shortBranches.at(el).first){
      position = el;
      return "s";
    }
  }
  for (unsigned int el=0; el<intBranches.size(); el++){
    if (branchname==intBranches.at(el).first){
      position = el;
      return "i";
    }
  }
  for (unsigned int el=0; el<floatBranches.size(); el++){
    if (branchname==floatBranches.at(el).first){
      position = el;
      return "f";
    }
  }
  return "unknown";
}
void HIPTwoBodyDecayAnalyzer::cleanBranches(){
  for (unsigned short el=0; el<shortBranches.size(); el++){
    if (shortBranches.at(el).second!=0) delete shortBranches.at(el).second;
    shortBranches.at(el).second=0;
  }
  shortBranches.clear();
  for (unsigned int el=0; el<intBranches.size(); el++){
    if (intBranches.at(el).second!=0) delete intBranches.at(el).second;
    intBranches.at(el).second=0;
  }
  intBranches.clear();
  for (unsigned int el=0; el<floatBranches.size(); el++){
    if (floatBranches.at(el).second!=0) delete floatBranches.at(el).second;
    floatBranches.at(el).second=0;
  }
  floatBranches.clear();
}
void HIPTwoBodyDecayAnalyzer::initializeBranches(){
  for (unsigned short el=0; el<shortBranches.size(); el++){
    if (shortBranches.at(el).second!=0) *(shortBranches.at(el).second)=0;
  }
  for (unsigned int el=0; el<intBranches.size(); el++){
    if (intBranches.at(el).second!=0) *(intBranches.at(el).second)=0;
  }
  for (unsigned int el=0; el<floatBranches.size(); el++){
    if (floatBranches.at(el).second!=0) *(floatBranches.at(el).second)=0;
  }
}

void HIPTwoBodyDecayAnalyzer::bookAllBranches(){
  const int nTrackTypes = 4;
  TString strTrackTypes[nTrackTypes]={
    "alcareco",
    "refit1",
    "refit2",
    "final"
  };
  for (unsigned int it=0; it<4; it++){
    TString strTrackType = strTrackTypes[it];
    bookBranch(strTrackType + "_present", "s");
    bookBranch(strTrackType + "_ZVtxFitOk", "s");
    bookBranch(strTrackType + "_ZMass", "f"); bookBranch(strTrackType + "_ZPt", "f"); bookBranch(strTrackType + "_ZPz", "f"); bookBranch(strTrackType + "_ZPhi", "f");
    bookBranch(strTrackType + "_ZVertex_x", "f"); bookBranch(strTrackType + "_ZVertex_y", "f"); bookBranch(strTrackType + "_ZVertex_z", "f"); bookBranch(strTrackType + "_ZVertex_NormChi2", "f");
    bookBranch(strTrackType + "_MuPlusVertex_x", "f"); bookBranch(strTrackType + "_MuPlusVertex_y", "f"); bookBranch(strTrackType + "_MuPlusVertex_z", "f");
    bookBranch(strTrackType + "_MuMinusPt_AfterZVtxFit", "f"); bookBranch(strTrackType + "_MuMinusPz_AfterZVtxFit", "f"); bookBranch(strTrackType + "_MuMinusPhi_AfterZVtxFit", "f");
    bookBranch(strTrackType + "_MuPlusPt_AfterZVtxFit", "f"); bookBranch(strTrackType + "_MuPlusPz_AfterZVtxFit", "f"); bookBranch(strTrackType + "_MuPlusPhi_AfterZVtxFit", "f");
    bookBranch(strTrackType + "_ZMass_AfterZVtxFit", "f"); bookBranch(strTrackType + "_ZPt_AfterZVtxFit", "f"); bookBranch(strTrackType + "_ZPz_AfterZVtxFit", "f"); bookBranch(strTrackType + "_ZPhi_AfterZVtxFit", "f");
    bookBranch(strTrackType + "_MuMinusPt", "f"); bookBranch(strTrackType + "_MuMinusPz", "f"); bookBranch(strTrackType + "_MuMinusPhi", "f");
    bookBranch(strTrackType + "_MuMinusVertex_x", "f"); bookBranch(strTrackType + "_MuMinusVertex_y", "f"); bookBranch(strTrackType + "_MuMinusVertex_z", "f");
    bookBranch(strTrackType + "_MuPlusPt", "f"); bookBranch(strTrackType + "_MuPlusPz", "f"); bookBranch(strTrackType + "_MuPlusPhi", "f");
  }
  actuateBranches();
}
bool HIPTwoBodyDecayAnalyzer::bookBranch(TString bname, TString btype){
  if (btype=="f") floatBranches.emplace_back(bname, new Float_t);
  else if (btype=="i") intBranches.emplace_back(bname, new Int_t);
  else if (btype=="s") shortBranches.emplace_back(bname, new Short_t);
  else{
    std::cerr << "HIPTwoBodyDecayAnalyzer::bookBranch: No support for type " << btype << " for the branch " << bname << " is available." << std::endl;
    return false;
  }
  return true;
}
bool HIPTwoBodyDecayAnalyzer::actuateBranches(){
  bool success=true;
  std::cout << "Begin HIPTwoBodyDecayAnalyzer::actuateBranches" << std::endl;
  std::cout << "Number of short branches: " << shortBranches.size() << std::endl;
  std::cout << "Number of int branches: " << intBranches.size() << std::endl;
  std::cout << "Number of float branches: " << floatBranches.size() << std::endl;
  if (tree!=0){
    for (unsigned short el=0; el<shortBranches.size(); el++){
      std::cout << "Actuating branch " << shortBranches.at(el).first << " at address " << shortBranches.at(el).second << std::endl;
      if (!tree->GetBranchStatus(shortBranches.at(el).first))
        tree->Branch(shortBranches.at(el).first, shortBranches.at(el).second);
      else std::cout << "Failed!" << std::endl;
    }
    for (unsigned int el=0; el<intBranches.size(); el++){
      std::cout << "Actuating branch " << intBranches.at(el).first << " at address " << intBranches.at(el).second << std::endl;
      if (!tree->GetBranchStatus(intBranches.at(el).first))
        tree->Branch(intBranches.at(el).first, intBranches.at(el).second);
      else std::cout << "Failed!" << std::endl;
    }
    for (unsigned int el=0; el<floatBranches.size(); el++){
      std::cout << "Actuating branch " << floatBranches.at(el).first << " at address " << floatBranches.at(el).second << std::endl;
      if (!tree->GetBranchStatus(floatBranches.at(el).first))
        tree->Branch(floatBranches.at(el).first, floatBranches.at(el).second);
      else std::cout << "Failed!" << std::endl;
    }
  }
  else success=false;
  if (!success) std::cerr << "HIPTwoBodyDecayAnalyzer::actuateBranch: Failed to actuate the branches!" << std::endl;
  return success;
}
HIPTwoBodyDecayAnalyzer::~HIPTwoBodyDecayAnalyzer(){
  cleanBranches();
}

void HIPTwoBodyDecayAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  using namespace edm;
  using namespace reco;
  using reco::TrackCollection;

  edm::Handle<reco::TrackCollection> alcarecotracks;
  iEvent.getByToken(alcareco_trackCollToken_, alcarecotracks);
  edm::Handle<reco::TrackCollection> refit1tracks;
  iEvent.getByToken(refit1_trackCollToken_, refit1tracks);
  edm::Handle<reco::TrackCollection> ctftracks;
  iEvent.getByToken(ctf_trackCollToken_, ctftracks);
  edm::Handle<reco::TrackCollection> finaltracks;
  iEvent.getByToken(final_trackCollToken_, finaltracks);

  edm::ESHandle<TransientTrackBuilder> theTTBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", theTTBuilder);

  initializeBranches();
  
  analyzeTrackCollection("alcareco", theTTBuilder, alcarecotracks);
  analyzeTrackCollection("refit1", theTTBuilder, refit1tracks);
  analyzeTrackCollection("refit2", theTTBuilder, ctftracks);
  analyzeTrackCollection("final", theTTBuilder, finaltracks);

  tree->Fill();
}

void HIPTwoBodyDecayAnalyzer::beginJob(){}
void HIPTwoBodyDecayAnalyzer::endJob(){}

void HIPTwoBodyDecayAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void HIPTwoBodyDecayAnalyzer::analyzeTrackCollection(TString strTrackType, edm::ESHandle<TransientTrackBuilder>& theTTBuilder, edm::Handle<reco::TrackCollection>& hTrackColl, bool verbose){
  if (verbose) std::cout << "Starting to process the track collection for " << strTrackType << std::endl;

  using namespace edm;
  using namespace reco;
  using reco::TrackCollection;

  if (!hTrackColl.isValid()){
    if (verbose) std::cout << "Track collection is invalid." << std::endl;
    return;
  }
  if (hTrackColl->size()<2){
    if (verbose) std::cout << "Track collection size<2." << std::endl;
    return;
  }

  unsigned int itrk=0;
  unsigned int j=0;
  int totalcharge=0;
  bool isValidPair=true;
  bool ZVtxOk=false;
  TLorentzVector trackMom[2];
  TLorentzVector trackMomAfterZVtxFit[2];
  TVector3 trackVtx[2];

  for (unsigned int jtrk=0; jtrk<2; jtrk++){
    trackMom[jtrk].SetXYZT(0, 0, 0, 0);
    trackVtx[jtrk].SetXYZ(0, 0, 0);
  }
  for (reco::TrackCollection::const_iterator track = hTrackColl->begin(); track != hTrackColl->end(); ++track){
    int charge = track->charge();
    totalcharge += charge;
    if (j==0){
      itrk = (charge>0 ? 1 : 0);
    }
    else itrk = 1-itrk;
    trackMom[itrk].SetPtEtaPhiM(track->pt(), track->eta(), track->phi(), 0.105);
    trackVtx[itrk].SetXYZ(track->vx(), track->vy(), track->vz());
    j++;
    if (j==2) break;
  }

  isValidPair = (totalcharge==0 && trackMom[0].P()!=0. && trackMom[1].P()!=0.);
  if (verbose && !isValidPair) std::cout << "Track collection does not contain a valid std::pair." << std::endl;
  setVal(strTrackType + "_present", (isValidPair ? (Short_t)1 : (Short_t)0));
  if (isValidPair){
    TLorentzVector ZMom = trackMom[0] + trackMom[1];
    setVal(strTrackType + "_ZPt", (Float_t)ZMom.Pt());
    setVal(strTrackType + "_ZPz", (Float_t)ZMom.Pz());
    setVal(strTrackType + "_ZPhi", (Float_t)ZMom.Phi());
    setVal(strTrackType + "_ZMass", (Float_t)ZMom.M());

    reco::Vertex ZVtx = fitDimuonVertex(theTTBuilder, hTrackColl, ZVtxOk);
    if (ZVtxOk){
      setVal(strTrackType + "_ZVertex_x", (Float_t)ZVtx.x());
      setVal(strTrackType + "_ZVertex_y", (Float_t)ZVtx.y());
      setVal(strTrackType + "_ZVertex_z", (Float_t)ZVtx.z());
      setVal(strTrackType + "_ZVertex_NormChi2", (Float_t)ZVtx.normalizedChi2());

      // Recalculate track momenta with this vertex as reference
      j=0;
      for (reco::TrackCollection::const_iterator track = hTrackColl->begin(); track != hTrackColl->end(); ++track){
        TransientTrack t_track = theTTBuilder->build(&(*track));
        AnalyticalImpactPointExtrapolator extrapolator(t_track.field());
        TrajectoryStateOnSurface closestIn3DSpaceState = extrapolator.extrapolate(t_track.impactPointState(), RecoVertex::convertPos(ZVtx.position()));
        GlobalVector mom = closestIn3DSpaceState.globalMomentum();
        int charge = track->charge();
        totalcharge += charge;
        if (j==0){
          itrk = (charge>0 ? 1 : 0);
        }
        else itrk = 1-itrk;
        trackMomAfterZVtxFit[itrk].SetXYZT(mom.x(), mom.y(), mom.z(), sqrt(pow(0.105, 2) + pow(mom.mag(), 2)));
        j++;
        if (j==2) break;
      }
      if (totalcharge!=0) std::cerr << "HIPTwoBodyDecayAnalyzer::analyzeTrackCollection: Something went wrong! The total charge is no longer 0!" << std::endl;
      for (unsigned int jtrk=0; jtrk<2; jtrk++){
        TString strMuCore = (jtrk==0 ? "MuMinus" : "MuPlus");
        setVal(strTrackType + "_" + strMuCore + "Pt_AfterZVtxFit", (Float_t)trackMomAfterZVtxFit[jtrk].Pt());
        setVal(strTrackType + "_" + strMuCore + "Pz_AfterZVtxFit", (Float_t)trackMomAfterZVtxFit[jtrk].Pz());
        setVal(strTrackType + "_" + strMuCore + "Phi_AfterZVtxFit", (Float_t)trackMomAfterZVtxFit[jtrk].Phi());
      }
      TLorentzVector ZMom_AfterZVtxFit = trackMomAfterZVtxFit[0] + trackMomAfterZVtxFit[1];
      setVal(strTrackType + "_ZPt_AfterZVtxFit", (Float_t)ZMom_AfterZVtxFit.Pt());
      setVal(strTrackType + "_ZPz_AfterZVtxFit", (Float_t)ZMom_AfterZVtxFit.Pz());
      setVal(strTrackType + "_ZPhi_AfterZVtxFit", (Float_t)ZMom_AfterZVtxFit.Phi());
      setVal(strTrackType + "_ZMass_AfterZVtxFit", (Float_t)ZMom_AfterZVtxFit.M());
    }
    else std::cerr << "HIPTwoBodyDecayAnalyzer::analyzeTrackCollection: Z vertex fit failed for track collection " << strTrackType << std::endl;
  }
  setVal(strTrackType + "_ZVtxFitOk", (ZVtxOk ? (Short_t)1 : (Short_t)0));
  for (unsigned int jtrk=0; jtrk<2; jtrk++){
    TString strMuCore = (jtrk==0 ? "MuMinus" : "MuPlus");
    setVal(strTrackType + "_" + strMuCore + "Pt", (Float_t)trackMom[jtrk].Pt());
    setVal(strTrackType + "_" + strMuCore + "Pz", (Float_t)trackMom[jtrk].Pz());
    setVal(strTrackType + "_" + strMuCore + "Phi", (Float_t)trackMom[jtrk].Phi());
    setVal(strTrackType + "_" + strMuCore + "Vertex_x", (Float_t)trackVtx[jtrk].X());
    setVal(strTrackType + "_" + strMuCore + "Vertex_y", (Float_t)trackVtx[jtrk].Y());
    setVal(strTrackType + "_" + strMuCore + "Vertex_z", (Float_t)trackVtx[jtrk].Z());
  }
}

reco::Vertex HIPTwoBodyDecayAnalyzer::fitDimuonVertex(edm::ESHandle<TransientTrackBuilder>& theTTBuilder, edm::Handle<reco::TrackCollection>& hTrackColl, bool& fitOk){
  using namespace edm;
  using namespace reco;

  std::vector<TransientTrack> t_tks;
  for (TrackCollection::const_iterator track = hTrackColl->begin(); track != hTrackColl->end(); ++track){
    TransientTrack tt = theTTBuilder->build(&(*track));
    t_tks.push_back(tt);
  }

  // Kalman vertex fit without constraint
  KalmanVertexFitter vtxFitter;
  TransientVertex stdVertex = vtxFitter.vertex(t_tks);
  fitOk = stdVertex.isValid();
  if (fitOk){
    reco::Vertex stdRecoVertex(
      Vertex::Point(stdVertex.position()), stdVertex.positionError().matrix(),
      stdVertex.totalChiSquared(), stdVertex.degreesOfFreedom(), 0
      );
    return stdRecoVertex;
  }
  else{
    reco::Vertex stdRecoVertex;
    return stdRecoVertex;
  }
}

DEFINE_FWK_MODULE(HIPTwoBodyDecayAnalyzer);
