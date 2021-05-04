#ifndef CommonTools_Puppi_PuppiProducer_h_
#define CommonTools_Puppi_PuppiProducer_h_
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "CommonTools/PileupAlgos/interface/PuppiContainer.h"
#include "CommonTools/PileupAlgos/interface/PuppiAlgo.h"

// ------------------------------------------------------------------------------------------
class PuppiProducer : public edm::stream::EDProducer<> {
public:
  explicit PuppiProducer(const edm::ParameterSet&);
  ~PuppiProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  typedef math::XYZTLorentzVector LorentzVector;
  typedef std::vector<LorentzVector> LorentzVectorCollection;
  typedef reco::VertexCollection VertexCollection;
  typedef edm::View<reco::Candidate> CandidateView;
  typedef std::vector<reco::PFCandidate> PFInputCollection;
  typedef std::vector<reco::PFCandidate> PFOutputCollection;
  typedef std::vector<pat::PackedCandidate> PackedOutputCollection;
  typedef edm::View<reco::PFCandidate> PFView;

private:
  virtual void beginJob();
  void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob();

  edm::EDGetTokenT<CandidateView> tokenPFCandidates_;
  edm::EDGetTokenT<VertexCollection> tokenVertices_;
  edm::EDGetTokenT<PuppiContainer> tokenPuppiContainer_;
  edm::EDGetTokenT<PFOutputCollection> tokenPuppiCandidates_;
  edm::EDGetTokenT<PackedOutputCollection> tokenPackedPuppiCandidates_;
  edm::EDGetTokenT<double> puProxyValueToken_;
  edm::EDPutTokenT<edm::ValueMap<float>> ptokenPupOut_;
  edm::EDPutTokenT<edm::ValueMap<LorentzVector>> ptokenP4PupOut_;
  edm::EDPutTokenT<edm::ValueMap<reco::CandidatePtr>> ptokenValues_;
  edm::EDPutTokenT<pat::PackedCandidateCollection> ptokenPackedPuppiCandidates_;
  edm::EDPutTokenT<reco::PFCandidateCollection> ptokenPuppiCandidates_;
  edm::EDPutTokenT<double> ptokenNalgos_;
  edm::EDPutTokenT<std::vector<double>> ptokenRawAlphas_;
  edm::EDPutTokenT<std::vector<double>> ptokenAlphas_;
  edm::EDPutTokenT<std::vector<double>> ptokenAlphasMed_;
  edm::EDPutTokenT<std::vector<double>> ptokenAlphasRms_;
  std::string fPuppiName;
  std::string fPFName;
  std::string fPVName;
  bool fPuppiDiagnostics;
  bool fPuppiNoLep;
  bool fUseFromPVLooseTight;
  bool fUseDZ;
  double fDZCut;
  double fEtaMinUseDZ;
  double fPtMaxCharged;
  double fEtaMaxCharged;
  double fPtMaxPhotons;
  double fEtaMaxPhotons;
  uint fNumOfPUVtxsForCharged;
  double fDZCutForChargedFromPUVtxs;
  bool fUseExistingWeights;
  bool fClonePackedCands;
  int fVtxNdofCut;
  double fVtxZCut;
  bool fUsePUProxyValue;
  std::unique_ptr<PuppiContainer> fPuppiContainer;
  std::vector<RecoObj> fRecoObjCollection;
};
#endif
