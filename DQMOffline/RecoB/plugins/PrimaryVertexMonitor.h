#ifndef DQMOffline_RecoB_PrimaryVertexMonitor_H
#define DQMOffline_RecoB_PrimaryVertexMonitor_H

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

/** \class PrimaryVertexMonitor
 *
 *
 */

namespace pvMonitor {
  // same logic used for the MTV:
  // cf https://github.com/cms-sw/cmssw/blob/master/Validation/RecoTrack/src/MTVHistoProducerAlgoForTracker.cc
  typedef dqm::reco::DQMStore DQMStore;

  void setBinLog(TAxis *axis) {
    int bins = axis->GetNbins();
    float from = axis->GetXmin();
    float to = axis->GetXmax();
    float width = (to - from) / bins;
    std::vector<float> new_bins(bins + 1, 0);
    for (int i = 0; i <= bins; i++) {
      new_bins[i] = TMath::Power(10, from + i * width);
    }
    axis->Set(bins, new_bins.data());
  }

  void setBinLogX(TH1 *h) {
    TAxis *axis = h->GetXaxis();
    setBinLog(axis);
  }
  void setBinLogY(TH1 *h) {
    TAxis *axis = h->GetYaxis();
    setBinLog(axis);
  }

  template <typename... Args>
  dqm::reco::MonitorElement *makeProfileIfLog(DQMStore::IBooker &ibook, bool logx, bool logy, Args &&...args) {
    auto prof = std::make_unique<TProfile>(std::forward<Args>(args)...);
    if (logx)
      setBinLogX(prof.get());
    if (logy)
      setBinLogY(prof.get());
    const auto &name = prof->GetName();
    return ibook.bookProfile(name, prof.release());
  }

  template <typename... Args>
  dqm::reco::MonitorElement *makeTH1IfLog(DQMStore::IBooker &ibook, bool logx, bool logy, Args &&...args) {
    auto h1 = std::make_unique<TH1F>(std::forward<Args>(args)...);
    if (logx)
      setBinLogX(h1.get());
    if (logy)
      setBinLogY(h1.get());
    const auto &name = h1->GetName();
    return ibook.book1D(name, h1.release());
  }

}  // namespace pvMonitor

class PrimaryVertexMonitor : public DQMEDAnalyzer {
public:
  explicit PrimaryVertexMonitor(const edm::ParameterSet &pSet);
  ~PrimaryVertexMonitor() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  struct IPMonitoring {
    std::string varname_;
    float pTcut_;
    dqm::reco::MonitorElement *IP_, *IPErr_, *IPPull_;
    dqm::reco::MonitorElement *IPVsPhi_, *IPVsEta_, *IPVsPt_;
    dqm::reco::MonitorElement *IPErrVsPhi_, *IPErrVsEta_, *IPErrVsPt_;
    dqm::reco::MonitorElement *IPVsEtaVsPhi_, *IPErrVsEtaVsPhi_;

    void bookIPMonitor(DQMStore::IBooker &, const edm::ParameterSet &);

  private:
    int PhiBin_, EtaBin_, PtBin_;
    double PhiMin_, PhiMax_, EtaMin_, EtaMax_, PtMin_, PtMax_;
  };

private:
  void pvTracksPlots(const reco::Vertex &v);
  void vertexPlots(const reco::Vertex &v, const reco::BeamSpot &beamSpot, int i);

  // event data

  const edm::InputTag vertexInputTag_;
  const edm::InputTag beamSpotInputTag_;
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  using VertexScore = edm::ValueMap<float>;
  const edm::EDGetTokenT<VertexScore> scoreToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;

  // configuration

  const edm::ParameterSet conf_;
  const std::string dqmLabel;
  const std::string TopFolderName_;
  const std::string AlignmentLabel_;
  const int ndof_;
  const bool useHPfoAlignmentPlots_;
  bool errorPrinted_;

  static constexpr int cmToUm = 10000;

  // the histos
  MonitorElement *nbvtx, *nbgvtx, *nbtksinvtx[2], *trksWeight[2], *score[2];
  MonitorElement *tt[2];
  MonitorElement *xrec[2], *yrec[2], *zrec[2], *xDiff[2], *yDiff[2], *xerr[2], *yerr[2], *zerr[2];
  MonitorElement *xerrVsTrks[2], *yerrVsTrks[2], *zerrVsTrks[2];
  MonitorElement *ntracksVsZ[2];
  MonitorElement *vtxchi2[2], *vtxndf[2], *vtxprob[2], *nans[2];
  MonitorElement *type[2];
  MonitorElement *bsX, *bsY, *bsZ, *bsSigmaZ, *bsDxdz, *bsDydz, *bsBeamWidthX, *bsBeamWidthY, *bsType;

  MonitorElement *sumpt, *ntracks, *weight, *chi2ndf, *chi2prob;
  MonitorElement *trackpt;
  MonitorElement *phi_pt1, *eta_pt1;
  MonitorElement *phi_pt10, *eta_pt10;

  MonitorElement *dxy2;

  // IP monitoring structs
  IPMonitoring dxy_pt1;
  IPMonitoring dxy_pt10;

  IPMonitoring dz_pt1;
  IPMonitoring dz_pt10;
};

#endif
