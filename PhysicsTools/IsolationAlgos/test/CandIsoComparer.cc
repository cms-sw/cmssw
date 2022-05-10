#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>

#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TH1.h>
#include <TProfile.h>

class CandIsoComparer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  CandIsoComparer(const edm::ParameterSet&);

  virtual ~CandIsoComparer();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

private:
  std::string label_;
  edm::EDGetTokenT<reco::CandViewDoubleAssociations> src1Token_;
  edm::EDGetTokenT<reco::CandViewDoubleAssociations> src2Token_;
  TH1 *h1_[2], *h2_[2], *hd_[2];
  TProfile *p1_[2], *p2_[2], *pd_[2];
};

/// constructor with config
CandIsoComparer::CandIsoComparer(const edm::ParameterSet& par)
    : src1Token_(consumes<reco::CandViewDoubleAssociations>(par.getParameter<edm::InputTag>("src1"))),
      src2Token_(consumes<reco::CandViewDoubleAssociations>(par.getParameter<edm::InputTag>("src2"))) {
  label_ = par.getParameter<std::string>("@module_label");

  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  double max = par.getParameter<double>("max");
  double rmax = par.getParameter<double>("rmax");
  double dmax = par.getParameter<double>("dmax");
  int32_t bins = par.getParameter<int32_t>("bins");

  h1_[0] = fs->make<TH1F>("first", "first", bins, 0, max);
  h2_[0] = fs->make<TH1F>("second", "second", bins, 0, max);
  hd_[0] = fs->make<TH1F>("diff", "diff", bins, -dmax, dmax);
  h1_[1] = fs->make<TH1F>("firstRel", "firstRel", bins, 0, rmax);
  h2_[1] = fs->make<TH1F>("secondRel", "secondRel", bins, 0, rmax);
  hd_[1] = fs->make<TH1F>("diffRel", "diffRel", bins, -dmax * rmax / max, dmax * rmax / max);

  p1_[0] = fs->make<TProfile>("firstEta", "firstEta", bins, -3.0, 3.0);
  p2_[0] = fs->make<TProfile>("secondEta", "secondEta", bins, -3.0, 3.0);
  pd_[0] = fs->make<TProfile>("diffEta", "diffEta", bins, -3.0, 3.0);
  p1_[1] = fs->make<TProfile>("firstPt", "firstPt", bins * 2, 0, 120.0);
  p2_[1] = fs->make<TProfile>("secondPt", "secondPt", bins * 2, 0, 120.0);
  pd_[1] = fs->make<TProfile>("diffPt", "diffPt", bins * 2, 0, 120.0);
}

/// destructor
CandIsoComparer::~CandIsoComparer() {}

void CandIsoComparer::endJob() { std::cout << "MODULE " << label_ << " RMS DIFF = " << hd_[0]->GetRMS() << std::endl; }

/// build deposits
void CandIsoComparer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace reco::isodeposit;
  edm::Handle<reco::CandViewDoubleAssociations> hDeps1, hDeps2;
  iEvent.getByToken(src1Token_, hDeps1);
  iEvent.getByToken(src2Token_, hDeps2);
  for (size_t dep = 0; dep < hDeps1->size(); ++dep) {
    const reco::CandidateBaseRef& cand = hDeps1->key(dep);
    double iso1 = (*hDeps1)[cand];
    double iso2 = (*hDeps2)[cand];
    h1_[0]->Fill(iso1);
    h2_[0]->Fill(iso2);
    hd_[0]->Fill(iso1 - iso2);
    h1_[1]->Fill(iso1 / cand->pt());
    h2_[1]->Fill(iso2 / cand->pt());
    hd_[1]->Fill((iso1 - iso2) / cand->pt());
    p1_[0]->Fill(cand->eta(), iso1);
    p2_[0]->Fill(cand->eta(), iso2);
    pd_[0]->Fill(cand->eta(), iso1 - iso2);
    p1_[1]->Fill(cand->pt(), iso1);
    p2_[1]->Fill(cand->pt(), iso2);
    pd_[1]->Fill(cand->pt(), iso1 - iso2);
  }
}

DEFINE_FWK_MODULE(CandIsoComparer);
