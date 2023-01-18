// -*- C++ -*-
// Package:    SiPixelCompareVertexSoA
// Class:      SiPixelCompareVertexSoA
//
/**\class SiPixelCompareVertexSoA SiPixelCompareVertexSoA.cc
*/
//
// Author: Suvankar Roy Chowdhury
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousHost.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class SiPixelCompareVertexSoA : public DQMEDAnalyzer {
public:
  using IndToEdm = std::vector<uint16_t>;
  explicit SiPixelCompareVertexSoA(const edm::ParameterSet&);
  ~SiPixelCompareVertexSoA() override = default;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<ZVertexSoAHost> tokenSoAVertexCPU_;
  const edm::EDGetTokenT<ZVertexSoAHost> tokenSoAVertexGPU_;
  const edm::EDGetTokenT<reco::BeamSpot> tokenBeamSpot_;
  const std::string topFolderName_;
  const float dzCut_;
  MonitorElement* hnVertex_;
  MonitorElement* hx_;
  MonitorElement* hy_;
  MonitorElement* hz_;
  MonitorElement* hchi2_;
  MonitorElement* hchi2oNdof_;
  MonitorElement* hptv2_;
  MonitorElement* hntrks_;
  MonitorElement* hxdiff_;
  MonitorElement* hydiff_;
  MonitorElement* hzdiff_;
};

//
// constructors
//

// Note tokenSoAVertexGPU_ contains data copied from device to host, hence is a HostCollection
SiPixelCompareVertexSoA::SiPixelCompareVertexSoA(const edm::ParameterSet& iConfig)
    : tokenSoAVertexCPU_(consumes<ZVertexSoAHost>(iConfig.getParameter<edm::InputTag>("pixelVertexSrcCPU"))),
      tokenSoAVertexGPU_(consumes<ZVertexSoAHost>(iConfig.getParameter<edm::InputTag>("pixelVertexSrcGPU"))),
      tokenBeamSpot_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotSrc"))),
      topFolderName_(iConfig.getParameter<std::string>("topFolderName")),
      dzCut_(iConfig.getParameter<double>("dzCut")) {}

//
// -- Analyze
//
void SiPixelCompareVertexSoA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& vsoaHandleCPU = iEvent.getHandle(tokenSoAVertexCPU_);
  const auto& vsoaHandleGPU = iEvent.getHandle(tokenSoAVertexGPU_);
  if (not vsoaHandleCPU or not vsoaHandleGPU) {
    edm::LogWarning out("SiPixelCompareVertexSoA");
    if (not vsoaHandleCPU) {
      out << "reference (cpu) tracks not found; ";
    }
    if (not vsoaHandleGPU) {
      out << "target (gpu) tracks not found; ";
    }
    out << "the comparison will not run.";
    return;
  }

  auto const& vsoaCPU = *vsoaHandleCPU;
  int nVerticesCPU = vsoaCPU.view().nvFinal();
  auto const& vsoaGPU = *vsoaHandleGPU;
  int nVerticesGPU = vsoaGPU.view().nvFinal();

  auto bsHandle = iEvent.getHandle(tokenBeamSpot_);
  float x0 = 0., y0 = 0., z0 = 0., dxdz = 0., dydz = 0.;
  if (!bsHandle.isValid()) {
    edm::LogWarning("SiPixelCompareVertexSoA") << "No beamspot found. returning vertexes with (0,0,Z) ";
  } else {
    const reco::BeamSpot& bs = *bsHandle;
    x0 = bs.x0();
    y0 = bs.y0();
    z0 = bs.z0();
    dxdz = bs.dxdz();
    dydz = bs.dydz();
  }

  for (int ivc = 0; ivc < nVerticesCPU; ivc++) {
    auto sic = vsoaCPU.view()[ivc].sortInd();
    auto zc = vsoaCPU.view()[sic].zv();
    auto xc = x0 + dxdz * zc;
    auto yc = y0 + dydz * zc;
    zc += z0;

    auto ndofCPU = vsoaCPU.view()[sic].ndof();
    auto chi2CPU = vsoaCPU.view()[sic].chi2();

    const int32_t notFound = -1;
    int32_t closestVtxidx = notFound;
    float mindz = dzCut_;

    for (int ivg = 0; ivg < nVerticesGPU; ivg++) {
      auto sig = vsoaGPU.view()[ivg].sortInd();
      auto zgc = vsoaGPU.view()[sig].zv() + z0;
      auto zDist = std::abs(zc - zgc);
      //insert some matching condition
      if (zDist > dzCut_)
        continue;
      if (mindz > zDist) {
        mindz = zDist;
        closestVtxidx = sig;
      }
    }
    if (closestVtxidx == notFound)
      continue;

    auto zg = vsoaGPU.view()[closestVtxidx].zv();
    auto xg = x0 + dxdz * zg;
    auto yg = y0 + dydz * zg;
    zg += z0;
    auto ndofGPU = vsoaGPU.view()[closestVtxidx].ndof();
    auto chi2GPU = vsoaGPU.view()[closestVtxidx].chi2();

    hx_->Fill(xc - x0, xg - x0);
    hy_->Fill(yc - y0, yg - y0);
    hz_->Fill(zc, zg);
    hxdiff_->Fill(xc - xg);
    hydiff_->Fill(yc - yg);
    hzdiff_->Fill(zc - zg);
    hchi2_->Fill(chi2CPU, chi2GPU);
    hchi2oNdof_->Fill(chi2CPU / ndofCPU, chi2GPU / ndofGPU);
    hptv2_->Fill(vsoaCPU.view()[sic].ptv2(), vsoaGPU.view()[closestVtxidx].ptv2());
    hntrks_->Fill(ndofCPU + 1, ndofGPU + 1);
  }
  hnVertex_->Fill(nVerticesCPU, nVerticesGPU);
}

//
// -- Book Histograms
//
void SiPixelCompareVertexSoA::bookHistograms(DQMStore::IBooker& ibooker,
                                             edm::Run const& iRun,
                                             edm::EventSetup const& iSetup) {
  ibooker.cd();
  ibooker.setCurrentFolder(topFolderName_);

  // FIXME: all the 2D correlation plots are quite heavy in terms of memory consumption, so a as soon as DQM supports either TH2I or THnSparse
  // these should be moved to a less resource consuming format
  hnVertex_ = ibooker.book2I("nVertex", "# of Vertices;CPU;GPU", 101, -0.5, 100.5, 101, -0.5, 100.5);
  hx_ = ibooker.book2I("vx", "Vertez x - Beamspot x;CPU;GPU", 50, -0.1, 0.1, 50, -0.1, 0.1);
  hy_ = ibooker.book2I("vy", "Vertez y - Beamspot y;CPU;GPU", 50, -0.1, 0.1, 50, -0.1, 0.1);
  hz_ = ibooker.book2I("vz", "Vertez z;CPU;GPU", 30, -30., 30., 30, -30., 30.);
  hchi2_ = ibooker.book2I("chi2", "Vertex chi-squared;CPU;GPU", 40, 0., 20., 40, 0., 20.);
  hchi2oNdof_ = ibooker.book2I("chi2oNdof", "Vertex chi-squared/Ndof;CPU;GPU", 40, 0., 20., 40, 0., 20.);
  hptv2_ = ibooker.book2I("ptsq", "Vertex #sum (p_{T})^{2};CPU;GPU", 200, 0., 200., 200, 0., 200.);
  hntrks_ = ibooker.book2I("ntrk", "#tracks associated;CPU;GPU", 100, -0.5, 99.5, 100, -0.5, 99.5);
  hntrks_ = ibooker.book2I("ntrk", "#tracks associated;CPU;GPU", 100, -0.5, 99.5, 100, -0.5, 99.5);
  hxdiff_ = ibooker.book1D("vxdiff", ";Vertex x difference (CPU - GPU);#entries", 100, -0.001, 0.001);
  hydiff_ = ibooker.book1D("vydiff", ";Vertex y difference (CPU - GPU);#entries", 100, -0.001, 0.001);
  hzdiff_ = ibooker.book1D("vzdiff", ";Vertex z difference (CPU - GPU);#entries", 100, -2.5, 2.5);
}

void SiPixelCompareVertexSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelVertexSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelVertexSrcCPU", edm::InputTag("pixelVerticesSoA@cpu"));
  desc.add<edm::InputTag>("pixelVertexSrcGPU", edm::InputTag("pixelVerticesSoA@cuda"));
  desc.add<edm::InputTag>("beamSpotSrc", edm::InputTag("offlineBeamSpot"));
  desc.add<std::string>("topFolderName", "SiPixelHeterogeneous/PixelVertexCompareSoAGPUvsCPU");
  desc.add<double>("dzCut", 1.);
  descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(SiPixelCompareVertexSoA);

using SiPixelPhase1CompareVertexSoA = SiPixelCompareVertexSoA;
DEFINE_FWK_MODULE(SiPixelPhase1CompareVertexSoA);
