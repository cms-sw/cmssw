// -*- C++ -*-
// Package:    SiPixelPhase1CompareVertexSoA
// Class:      SiPixelPhase1CompareVertexSoA
//
/**\class SiPixelPhase1CompareVertexSoA SiPixelPhase1CompareVertexSoA.cc 
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
#include "CUDADataFormats/Vertex/interface/ZVertexHeterogeneous.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class SiPixelPhase1CompareVertexSoA : public DQMEDAnalyzer {
public:
  using IndToEdm = std::vector<uint16_t>;
  explicit SiPixelPhase1CompareVertexSoA(const edm::ParameterSet&);
  ~SiPixelPhase1CompareVertexSoA() override = default;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<ZVertexHeterogeneous> tokenSoAVertexCPU_;
  const edm::EDGetTokenT<ZVertexHeterogeneous> tokenSoAVertexGPU_;
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
};

//
// constructors
//

SiPixelPhase1CompareVertexSoA::SiPixelPhase1CompareVertexSoA(const edm::ParameterSet& iConfig)
    : tokenSoAVertexCPU_(consumes<ZVertexHeterogeneous>(iConfig.getParameter<edm::InputTag>("pixelVertexSrcCPU"))),
      tokenSoAVertexGPU_(consumes<ZVertexHeterogeneous>(iConfig.getParameter<edm::InputTag>("pixelVertexSrcGPU"))),
      tokenBeamSpot_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotSrc"))),
      topFolderName_(iConfig.getParameter<std::string>("topFolderName")),
      dzCut_(iConfig.getParameter<double>("dzCut")) {}

//
// -- Analyze
//
void SiPixelPhase1CompareVertexSoA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& vsoaHandleCPU = iEvent.getHandle(tokenSoAVertexCPU_);
  const auto& vsoaHandleGPU = iEvent.getHandle(tokenSoAVertexGPU_);
  if (not vsoaHandleCPU or not vsoaHandleGPU) {
    edm::LogWarning out("SiPixelPhase1CompareTrackSoA");
    if (not vsoaHandleCPU) {
      out << "reference (cpu) tracks not found; ";
    }
    if (not vsoaHandleGPU) {
      out << "target (gpu) tracks not found; ";
    }
    out << "the comparison will not run.";
    return;
  }

  auto const& vsoaCPU = *vsoaHandleCPU->get();
  int nVerticesCPU = vsoaCPU.nvFinal;
  auto const& vsoaGPU = *vsoaHandleGPU->get();
  int nVerticesGPU = vsoaGPU.nvFinal;

  auto bsHandle = iEvent.getHandle(tokenBeamSpot_);
  float x0 = 0., y0 = 0., z0 = 0., dxdz = 0., dydz = 0.;
  if (!bsHandle.isValid()) {
    edm::LogWarning("PixelVertexProducer") << "No beamspot found. returning vertexes with (0,0,Z) ";
  } else {
    const reco::BeamSpot& bs = *bsHandle;
    x0 = bs.x0();
    y0 = bs.y0();
    z0 = bs.z0();
    dxdz = bs.dxdz();
    dydz = bs.dydz();
  }

  for (int ivc = 0; ivc < nVerticesCPU; ivc++) {
    auto sic = vsoaCPU.sortInd[ivc];
    auto zc = vsoaCPU.zv[sic];
    auto xc = x0 + dxdz * zc;
    auto yc = y0 + dydz * zc;
    zc += z0;

    auto ndofCPU = vsoaCPU.ndof[sic];
    auto chi2CPU = vsoaCPU.chi2[sic];

    const int32_t notFound = -1;
    int32_t closestVtxidx = notFound;
    float mindz = dzCut_;

    for (int ivg = 0; ivg < nVerticesGPU; ivg++) {
      auto sig = vsoaGPU.sortInd[ivg];
      auto zgc = vsoaGPU.zv[sig] + z0;
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

    auto zg = vsoaGPU.zv[closestVtxidx];
    auto xg = x0 + dxdz * zg;
    auto yg = y0 + dydz * zg;
    zg += z0;
    auto ndofGPU = vsoaGPU.ndof[closestVtxidx];
    auto chi2GPU = vsoaGPU.chi2[closestVtxidx];

    hx_->Fill(xc, xg);
    hy_->Fill(yc, yg);
    hz_->Fill(zc, zg);
    hchi2_->Fill(chi2CPU, chi2GPU);
    hchi2oNdof_->Fill(chi2CPU / ndofCPU, chi2GPU / ndofGPU);
    hptv2_->Fill(vsoaCPU.ptv2[sic], vsoaGPU.ptv2[closestVtxidx]);
    hntrks_->Fill(ndofCPU + 1, ndofGPU + 1);
  }
  hnVertex_->Fill(nVerticesCPU, nVerticesGPU);
}

//
// -- Book Histograms
//
void SiPixelPhase1CompareVertexSoA::bookHistograms(DQMStore::IBooker& ibooker,
                                                   edm::Run const& iRun,
                                                   edm::EventSetup const& iSetup) {
  //std::string top_folder = ""//
  ibooker.cd();
  ibooker.setCurrentFolder(topFolderName_);

  // FIXME: all the 2D correlation plots are quite heavy in terms of memory consumption, so a as soon as DQM supports either TH2I or THnSparse
  // these should be moved to a less resource consuming format
  hnVertex_ = ibooker.book2D("nVertex", "# of Vertex;CPU;GPU", 101, -0.5, 100.5, 101, -0.5, 100.5);
  hx_ = ibooker.book2D("vx", "Vertez x;CPU;GPU", 20, -0.1, 0.1, 20, -0.1, 0.1);
  hy_ = ibooker.book2D("vy", "Vertez y;CPU;GPU", 20, -0.1, 0.1, 20, -0.1, 0.1);
  hz_ = ibooker.book2D("vz", "Vertez z;CPU;GPU", 30, -30., 30., 30, -30., 30.);
  hchi2_ = ibooker.book2D("chi2", "Vertex chi-squared;CPU;GPU", 40, 0., 20., 40, 0., 20.);
  hchi2oNdof_ = ibooker.book2D("chi2oNdof", "Vertex chi-squared/Ndof;CPU;GPU", 40, 0., 20., 40, 0., 20.);
  hptv2_ = ibooker.book2D("ptsq", "Vertex p_T squared;CPU;GPU", 200, 0., 200., 200, 0., 200.);
  hntrks_ = ibooker.book2D("ntrk", "#tracks associated;CPU;GPU", 100, -0.5, 99.5, 100, -0.5, 99.5);
  hntrks_ = ibooker.book2D("ntrk", "#tracks associated;CPU;GPU", 100, -0.5, 99.5, 100, -0.5, 99.5);
}

void SiPixelPhase1CompareVertexSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelVertexSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelVertexSrcCPU", edm::InputTag("pixelVerticesSoA@cpu"));
  desc.add<edm::InputTag>("pixelVertexSrcGPU", edm::InputTag("pixelVerticesSoA@cuda"));
  desc.add<edm::InputTag>("beamSpotSrc", edm::InputTag("offlineBeamSpot"));
  desc.add<std::string>("topFolderName", "SiPixelHeterogeneous/PixelVertexCompareSoAGPUvsCPU");
  desc.add<double>("dzCut", 1.);
  descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(SiPixelPhase1CompareVertexSoA);
