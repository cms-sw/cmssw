#include "DQMOffline/PFTau/interface/MatchCandidateBenchmark.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TProfile.h>
#include <TROOT.h>

using namespace std;

MatchCandidateBenchmark::MatchCandidateBenchmark(Mode mode) : Benchmark(mode) {
  delta_et_Over_et_VS_et_ = nullptr;
  delta_et_VS_et_ = nullptr;
  delta_eta_VS_et_ = nullptr;
  delta_phi_VS_et_ = nullptr;

  BRdelta_et_Over_et_VS_et_ = nullptr;
  ERdelta_et_Over_et_VS_et_ = nullptr;
  // pTRes are initialzied in the setup since ptBinsPS.size() is needed

  histogramBooked_ = false;
}

MatchCandidateBenchmark::~MatchCandidateBenchmark() {}

void MatchCandidateBenchmark::setup(DQMStore::IBooker &b) {
  if (!histogramBooked_) {
    PhaseSpace ptPS;
    PhaseSpace dptOvptPS;
    PhaseSpace dptPS;
    PhaseSpace detaPS;
    PhaseSpace dphiPS;
    switch (mode_) {
      case VALIDATION:
        ptPS = PhaseSpace(100, 0, 1000);
        dptOvptPS = PhaseSpace(200, -1, 1);
        dphiPS = PhaseSpace(200, -1, 1);
        detaPS = PhaseSpace(200, -1, 1);
        dptPS = PhaseSpace(100, -100, 100);
        break;
      case DQMOFFLINE:
      default:
        ptPS = PhaseSpace(50, 0, 100);
        dptOvptPS = PhaseSpace(50, -1, 1);
        dphiPS = PhaseSpace(50, -1, 1);
        detaPS = PhaseSpace(50, -1, 1);
        dptPS = PhaseSpace(50, -50, 50);
        break;
    }
    float ptBins[11] = {0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 1000};
    int size = sizeof(ptBins) / sizeof(*ptBins);

    delta_et_Over_et_VS_et_ = book2D(b,
                                     "delta_et_Over_et_VS_et_",
                                     ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
                                     size,
                                     ptBins,
                                     dptOvptPS.n,
                                     dptOvptPS.m,
                                     dptOvptPS.M);

    BRdelta_et_Over_et_VS_et_ = book2D(b,
                                       "BRdelta_et_Over_et_VS_et_",
                                       ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
                                       size,
                                       ptBins,
                                       dptOvptPS.n,
                                       dptOvptPS.m,
                                       dptOvptPS.M);
    ERdelta_et_Over_et_VS_et_ = book2D(b,
                                       "ERdelta_et_Over_et_VS_et_",
                                       ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
                                       size,
                                       ptBins,
                                       dptOvptPS.n,
                                       dptOvptPS.m,
                                       dptOvptPS.M);

    delta_et_VS_et_ =
        book2D(b, "delta_et_VS_et_", ";E_{T, true} (GeV);#DeltaE_{T}", size, ptBins, dptPS.n, dptPS.m, dptPS.M);

    delta_eta_VS_et_ =
        book2D(b, "delta_eta_VS_et_", ";#E_{T, true} (GeV);#Delta#eta", size, ptBins, detaPS.n, detaPS.m, detaPS.M);

    delta_phi_VS_et_ =
        book2D(b, "delta_phi_VS_et_", ";E_{T, true} (GeV);#Delta#phi", size, ptBins, dphiPS.n, dphiPS.m, dphiPS.M);
    pTRes_.resize(size);
    BRpTRes_.resize(size);
    ERpTRes_.resize(size);
    for (size_t i = 0; i < pTRes_.size(); i++) {
      pTRes_[i] = nullptr;
      BRpTRes_[i] = nullptr;
      ERpTRes_[i] = nullptr;
    }

    histogramBooked_ = true;
  }
}

void MatchCandidateBenchmark::computePtBins(const edm::ParameterSet &ps, const edm::ParameterSet &ptPS) {
  const std::vector<double> &ptBinsPS = ps.getParameter<std::vector<double> >("VariablePtBins");
  if (ptBinsPS.size() > 1) {
    ptBins_.reserve(ptBinsPS.size());
    for (size_t i = 0; i < ptBinsPS.size(); i++)
      ptBins_.push_back(ptBinsPS[i]);
  } else {
    Int_t nFixedBins = ptPS.getParameter<int32_t>("nBin");
    ptBins_.reserve(nFixedBins + 1);
    for (Int_t i = 0; i <= nFixedBins; i++)
      ptBins_.push_back(ptPS.getParameter<double>("xMin") +
                        i * ((ptPS.getParameter<double>("xMax") - ptPS.getParameter<double>("xMin")) / nFixedBins));
  }
}
void MatchCandidateBenchmark::setup(DQMStore::IBooker &b, const edm::ParameterSet &parameterSet) {
  if (!histogramBooked_) {
    edm::ParameterSet ptPS = parameterSet.getParameter<edm::ParameterSet>("PtHistoParameter");
    edm::ParameterSet dptPS = parameterSet.getParameter<edm::ParameterSet>("DeltaPtHistoParameter");
    edm::ParameterSet dptOvptPS = parameterSet.getParameter<edm::ParameterSet>("DeltaPtOvPtHistoParameter");
    edm::ParameterSet detaPS = parameterSet.getParameter<edm::ParameterSet>("DeltaEtaHistoParameter");
    edm::ParameterSet dphiPS = parameterSet.getParameter<edm::ParameterSet>("DeltaPhiHistoParameter");
    computePtBins(parameterSet, ptPS);
    pTRes_.resize(ptBins_.size() - 1);
    BRpTRes_.resize(ptBins_.size() - 1);
    ERpTRes_.resize(ptBins_.size() - 1);
    if (!pTRes_.empty()) {
      for (size_t i = 0; i < pTRes_.size(); i++) {
        pTRes_[i] = nullptr;
        BRpTRes_[i] = nullptr;
        ERpTRes_[i] = nullptr;
      }
    }

    if (dptOvptPS.getParameter<bool>("switchOn")) {
      delta_et_Over_et_VS_et_ = book2D(b,
                                       "delta_et_Over_et_VS_et_",
                                       ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
                                       ptBins_.size() - 1,
                                       &(ptBins_.front()),
                                       dptOvptPS.getParameter<int32_t>("nBin"),
                                       dptOvptPS.getParameter<double>("xMin"),
                                       dptOvptPS.getParameter<double>("xMax"));
    }
    if (dptOvptPS.getParameter<bool>("slicingOn")) {
      for (size_t i = 0; i < pTRes_.size(); i++) {
        pTRes_[i] = book1D(b,
                           TString::Format("Pt%d_%d", (int)ptBins_[i], (int)ptBins_[i + 1]),
                           ";#Deltap_{T}/p_{T};Entries",
                           dptOvptPS.getParameter<int32_t>("nBin"),
                           dptOvptPS.getParameter<double>("xMin"),
                           dptOvptPS.getParameter<double>("xMax"));
        BRpTRes_[i] = book1D(b,
                             TString::Format("BRPt%d_%d", (int)ptBins_[i], (int)ptBins_[i + 1]),
                             ";#Deltap_{T}/p_{T};Entries",
                             dptOvptPS.getParameter<int32_t>("nBin"),
                             dptOvptPS.getParameter<double>("xMin"),
                             dptOvptPS.getParameter<double>("xMax"));
        ERpTRes_[i] = book1D(b,
                             TString::Format("ERPt%d_%d", (int)ptBins_[i], (int)ptBins_[i + 1]),
                             ";#Deltap_{T}/p_{T};Entries",
                             dptOvptPS.getParameter<int32_t>("nBin"),
                             dptOvptPS.getParameter<double>("xMin"),
                             dptOvptPS.getParameter<double>("xMax"));
      }
    }
    if (dptOvptPS.getParameter<bool>("BROn")) {
      BRdelta_et_Over_et_VS_et_ = book2D(b,
                                         "BRdelta_et_Over_et_VS_et_",
                                         ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
                                         ptBins_.size() - 1,
                                         &(ptBins_.front()),
                                         dptOvptPS.getParameter<int32_t>("nBin"),
                                         dptOvptPS.getParameter<double>("xMin"),
                                         dptOvptPS.getParameter<double>("xMax"));
    }
    if (dptOvptPS.getParameter<bool>("EROn")) {
      ERdelta_et_Over_et_VS_et_ = book2D(b,
                                         "ERdelta_et_Over_et_VS_et_",
                                         ";E_{T, true} (GeV);#DeltaE_{T}/E_{T}",
                                         ptBins_.size() - 1,
                                         &(ptBins_.front()),
                                         dptOvptPS.getParameter<int32_t>("nBin"),
                                         dptOvptPS.getParameter<double>("xMin"),
                                         dptOvptPS.getParameter<double>("xMax"));
    }

    if (dptPS.getParameter<bool>("switchOn")) {
      delta_et_VS_et_ = book2D(b,
                               "delta_et_VS_et_",
                               ";E_{T, true} (GeV);#DeltaE_{T}",
                               ptBins_.size() - 1,
                               &(ptBins_.front()),
                               dptPS.getParameter<int32_t>("nBin"),
                               dptPS.getParameter<double>("xMin"),
                               dptPS.getParameter<double>("xMax"));
    }

    if (detaPS.getParameter<bool>("switchOn")) {
      delta_eta_VS_et_ = book2D(b,
                                "delta_eta_VS_et_",
                                ";E_{T, true} (GeV);#Delta#eta",
                                ptBins_.size() - 1,
                                &(ptBins_.front()),
                                detaPS.getParameter<int32_t>("nBin"),
                                detaPS.getParameter<double>("xMin"),
                                detaPS.getParameter<double>("xMax"));
    }

    if (dphiPS.getParameter<bool>("switchOn")) {
      delta_phi_VS_et_ = book2D(b,
                                "delta_phi_VS_et_",
                                ";E_{T, true} (GeV);#Delta#phi",
                                ptBins_.size() - 1,
                                &(ptBins_.front()),
                                dphiPS.getParameter<int32_t>("nBin"),
                                dphiPS.getParameter<double>("xMin"),
                                dphiPS.getParameter<double>("xMax"));
    }
    eta_min_barrel_ = dptOvptPS.getParameter<double>("BREtaMin");
    eta_max_barrel_ = dptOvptPS.getParameter<double>("BREtaMax");
    eta_min_endcap_ = dptOvptPS.getParameter<double>("EREtaMin");
    eta_max_endcap_ = dptOvptPS.getParameter<double>("EREtaMax");
    histogramBooked_ = true;
  }
}

void MatchCandidateBenchmark::fillOne(const reco::Candidate &cand, const reco::Candidate &matchedCand) {
  if (!isInRange(cand.pt(), cand.eta(), cand.phi()))
    return;

  if (histogramBooked_) {
    if (delta_et_Over_et_VS_et_)
      delta_et_Over_et_VS_et_->Fill(matchedCand.pt(), (cand.pt() - matchedCand.pt()) / matchedCand.pt());
    if (fabs(cand.eta()) <= 1.4)
      if (BRdelta_et_Over_et_VS_et_)
        BRdelta_et_Over_et_VS_et_->Fill(matchedCand.pt(), (cand.pt() - matchedCand.pt()) / matchedCand.pt());
    if (fabs(cand.eta()) >= 1.6 && fabs(cand.eta()) <= 2.4)
      if (ERdelta_et_Over_et_VS_et_)
        ERdelta_et_Over_et_VS_et_->Fill(matchedCand.pt(), (cand.pt() - matchedCand.pt()) / matchedCand.pt());
    if (delta_et_VS_et_)
      delta_et_VS_et_->Fill(matchedCand.pt(), cand.pt() - matchedCand.pt());
    if (delta_eta_VS_et_)
      delta_eta_VS_et_->Fill(matchedCand.pt(), cand.eta() - matchedCand.eta());
    if (delta_phi_VS_et_)
      delta_phi_VS_et_->Fill(matchedCand.pt(), cand.phi() - matchedCand.phi());
  }
}

bool MatchCandidateBenchmark::inEtaRange(double value, bool inBarrel) {
  if (inBarrel) {
    return std::abs(value) >= eta_min_barrel_ && std::abs(value) <= eta_max_barrel_;
  }
  return std::abs(value) >= eta_min_endcap_ && std::abs(value) <= eta_max_endcap_;
}

void MatchCandidateBenchmark::fillOne(const reco::Candidate &cand,
                                      const reco::Candidate &matchedCand,
                                      const edm::ParameterSet &parameterSet) {
  if (!isInRange(cand.pt(), cand.eta(), cand.phi()))
    return;

  if (histogramBooked_) {
    edm::ParameterSet dptOvptPS = parameterSet.getParameter<edm::ParameterSet>("DeltaPtOvPtHistoParameter");

    if (matchedCand.pt() > ptBins_.at(0)) {  // underflow problem
      if (delta_et_Over_et_VS_et_)
        delta_et_Over_et_VS_et_->Fill(matchedCand.pt(), (cand.pt() - matchedCand.pt()) / matchedCand.pt());
      if (BRdelta_et_Over_et_VS_et_ and inBarrelRange(matchedCand.eta()))
        BRdelta_et_Over_et_VS_et_->Fill(matchedCand.pt(), (cand.pt() - matchedCand.pt()) / matchedCand.pt());
      if (ERdelta_et_Over_et_VS_et_ and inEndcapRange(matchedCand.eta()))
        ERdelta_et_Over_et_VS_et_->Fill(matchedCand.pt(), (cand.pt() - matchedCand.pt()) / matchedCand.pt());
      if (delta_et_VS_et_)
        delta_et_VS_et_->Fill(matchedCand.pt(), cand.pt() - matchedCand.pt());
      if (delta_eta_VS_et_)
        delta_eta_VS_et_->Fill(matchedCand.pt(), cand.eta() - matchedCand.eta());
      if (delta_phi_VS_et_)
        delta_phi_VS_et_->Fill(matchedCand.pt(), cand.phi() - matchedCand.phi());
    }

    for (size_t i = 0; i < pTRes_.size(); i++) {
      if (matchedCand.pt() >= ptBins_.at(i) && matchedCand.pt() < ptBins_.at(i + 1)) {
        if (pTRes_[i])
          pTRes_[i]->Fill((cand.pt() - matchedCand.pt()) / matchedCand.pt());
        if (BRpTRes_[i])
          BRpTRes_[i]->Fill((cand.pt() - matchedCand.pt()) / matchedCand.pt());  // Fill Barrel
        if (ERpTRes_[i])
          ERpTRes_[i]->Fill((cand.pt() - matchedCand.pt()) / matchedCand.pt());  // Fill Endcap
      }
    }
  }
}
