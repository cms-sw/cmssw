// TODO: change file name to SiPixelCompareVerticesSoA.cc when CUDA code is removed

// -*- C++ -*-
// Package:    SiPixelCompareVertices
// Class:      SiPixelCompareVertices
//
/**\class SiPixelCompareVertices SiPixelCompareVertices.cc
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
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousHost.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

// TODO: change class name to SiPixelCompareVerticesSoA when CUDA code is removed
class SiPixelCompareVertices : public DQMEDAnalyzer {
public:
  using IndToEdm = std::vector<uint16_t>;
  explicit SiPixelCompareVertices(const edm::ParameterSet&);
  ~SiPixelCompareVertices() override = default;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  // analyzeSeparate is templated to accept distinct types of SoAs
  // The default use case is to use vertices from Alpaka reconstructed on CPU and GPU;
  template <typename U, typename V>
  void analyzeSeparate(U tokenRef, V tokenTar, const edm::Event& iEvent);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // these two are both on Host but originally they have been produced on Host or on Device
  const edm::EDGetTokenT<ZVertexHost> tokenSoAVertexReferenceSoA_;
  const edm::EDGetTokenT<ZVertexHost> tokenSoAVertexTargetSoA_;
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

SiPixelCompareVertices::SiPixelCompareVertices(const edm::ParameterSet& iConfig)
    : tokenSoAVertexReferenceSoA_(
          consumes<ZVertexHost>(iConfig.getParameter<edm::InputTag>("pixelVertexReferenceSoA"))),
      tokenSoAVertexTargetSoA_(consumes<ZVertexHost>(iConfig.getParameter<edm::InputTag>("pixelVertexTargetSoA"))),
      tokenBeamSpot_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotSrc"))),
      topFolderName_(iConfig.getParameter<std::string>("topFolderName")),
      dzCut_(iConfig.getParameter<double>("dzCut")) {}

template <typename U, typename V>
void SiPixelCompareVertices::analyzeSeparate(U tokenRef, V tokenTar, const edm::Event& iEvent) {
  const auto& vsoaHandleRef = iEvent.getHandle(tokenRef);
  const auto& vsoaHandleTar = iEvent.getHandle(tokenTar);

  if (not vsoaHandleRef or not vsoaHandleTar) {
    edm::LogWarning out("SiPixelCompareVertices");
    if (not vsoaHandleRef) {
      out << "reference vertices not found; ";
    }
    if (not vsoaHandleTar) {
      out << "Refget vertices not found; ";
    }
    out << "the comparison will not run.";
    return;
  }

  auto const& vsoaRef = *vsoaHandleRef;
  int nVerticesRef = vsoaRef.view().nvFinal();
  auto const& vsoaTar = *vsoaHandleTar;
  int nVerticesTar = vsoaTar.view().nvFinal();

  auto bsHandle = iEvent.getHandle(tokenBeamSpot_);
  float x0 = 0., y0 = 0., z0 = 0., dxdz = 0., dydz = 0.;
  if (!bsHandle.isValid()) {
    edm::LogWarning("SiPixelCompareVertices") << "No beamspot found. returning vertexes with (0,0,Z) ";
  } else {
    const reco::BeamSpot& bs = *bsHandle;
    x0 = bs.x0();
    y0 = bs.y0();
    z0 = bs.z0();
    dxdz = bs.dxdz();
    dydz = bs.dydz();
  }

  for (int ivc = 0; ivc < nVerticesRef; ivc++) {
    auto sic = vsoaRef.view()[ivc].sortInd();
    auto zc = vsoaRef.view()[sic].zv();
    auto xc = x0 + dxdz * zc;
    auto yc = y0 + dydz * zc;
    zc += z0;

    auto ndofRef = vsoaRef.view()[sic].ndof();
    auto chi2Ref = vsoaRef.view()[sic].chi2();

    const int32_t notFound = -1;
    int32_t closestVtxidx = notFound;
    float mindz = dzCut_;

    for (int ivg = 0; ivg < nVerticesTar; ivg++) {
      auto sig = vsoaTar.view()[ivg].sortInd();
      auto zgc = vsoaTar.view()[sig].zv() + z0;
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

    auto zg = vsoaTar.view()[closestVtxidx].zv();
    auto xg = x0 + dxdz * zg;
    auto yg = y0 + dydz * zg;
    zg += z0;
    auto ndofTar = vsoaTar.view()[closestVtxidx].ndof();
    auto chi2Tar = vsoaTar.view()[closestVtxidx].chi2();

    hx_->Fill(xc - x0, xg - x0);
    hy_->Fill(yc - y0, yg - y0);
    hz_->Fill(zc, zg);
    hxdiff_->Fill(xc - xg);
    hydiff_->Fill(yc - yg);
    hzdiff_->Fill(zc - zg);
    hchi2_->Fill(chi2Ref, chi2Tar);
    hchi2oNdof_->Fill(chi2Ref / ndofRef, chi2Tar / ndofTar);
    hptv2_->Fill(vsoaRef.view()[sic].ptv2(), vsoaTar.view()[closestVtxidx].ptv2());
    hntrks_->Fill(ndofRef + 1, ndofTar + 1);
  }
  hnVertex_->Fill(nVerticesRef, nVerticesTar);
}

//
// -- Analyze
//
void SiPixelCompareVertices::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // The default use case is to use vertices from Alpaka reconstructed on CPU and GPU;
  // The function is left templated if any other cases need to be added
  analyzeSeparate(tokenSoAVertexReferenceSoA_, tokenSoAVertexTargetSoA_, iEvent);
}

//
// -- Book Histograms
//
void SiPixelCompareVertices::bookHistograms(DQMStore::IBooker& ibooker,
                                            edm::Run const& iRun,
                                            edm::EventSetup const& iSetup) {
  ibooker.cd();
  ibooker.setCurrentFolder(topFolderName_);

  // FIXME: all the 2D correlation plots are quite heavy in terms of memory consumption, so a as soon as DQM supports either TH2I or THnSparse
  // these should be moved to a less resource consuming format
  hnVertex_ = ibooker.book2I("nVertex", "# of Vertices;Reference;Target", 101, -0.5, 100.5, 101, -0.5, 100.5);
  hx_ = ibooker.book2I("vx", "Vertez x - Beamspot x;Reference;Target", 50, -0.1, 0.1, 50, -0.1, 0.1);
  hy_ = ibooker.book2I("vy", "Vertez y - Beamspot y;Reference;Target", 50, -0.1, 0.1, 50, -0.1, 0.1);
  hz_ = ibooker.book2I("vz", "Vertez z;Reference;Target", 30, -30., 30., 30, -30., 30.);
  hchi2_ = ibooker.book2I("chi2", "Vertex chi-squared;Reference;Target", 40, 0., 20., 40, 0., 20.);
  hchi2oNdof_ = ibooker.book2I("chi2oNdof", "Vertex chi-squared/Ndof;Reference;Target", 40, 0., 20., 40, 0., 20.);
  hptv2_ = ibooker.book2I("ptsq", "Vertex #sum (p_{T})^{2};Reference;Target", 200, 0., 200., 200, 0., 200.);
  hntrks_ = ibooker.book2I("ntrk", "#tracks associated;Reference;Target", 100, -0.5, 99.5, 100, -0.5, 99.5);
  hntrks_ = ibooker.book2I("ntrk", "#tracks associated;Reference;Target", 100, -0.5, 99.5, 100, -0.5, 99.5);
  hxdiff_ = ibooker.book1D("vxdiff", ";Vertex x difference (Reference - Target);#entries", 100, -0.001, 0.001);
  hydiff_ = ibooker.book1D("vydiff", ";Vertex y difference (Reference - Target);#entries", 100, -0.001, 0.001);
  hzdiff_ = ibooker.book1D("vzdiff", ";Vertex z difference (Reference - Target);#entries", 100, -2.5, 2.5);
}

void SiPixelCompareVertices::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelVertexSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelVertexReferenceSoA", edm::InputTag("pixelVerticesAlpakaSerial"));
  desc.add<edm::InputTag>("pixelVertexTargetSoA", edm::InputTag("pixelVerticesAlpaka"));
  desc.add<edm::InputTag>("beamSpotSrc", edm::InputTag("offlineBeamSpot"));
  desc.add<std::string>("topFolderName", "SiPixelHeterogeneous/PixelVertexCompareSoADeviceVSHost");
  desc.add<double>("dzCut", 1.);
  descriptions.addWithDefaultLabel(desc);
}

// TODO: change module name to SiPixelCompareVerticesSoA when CUDA code is removed
DEFINE_FWK_MODULE(SiPixelCompareVertices);
