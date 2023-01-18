// -*- C++ -*-
///bookLayer
// Package:    SiPixelMonitorVertexSoA
// Class:      SiPixelMonitorVertexSoA
//
/**\class SiPixelMonitorVertexSoA SiPixelMonitorVertexSoA.cc
*/
//
// Author: Suvankar Roy Chowdhury
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousHost.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class SiPixelMonitorVertexSoA : public DQMEDAnalyzer {
public:
  using IndToEdm = std::vector<uint16_t>;
  explicit SiPixelMonitorVertexSoA(const edm::ParameterSet&);
  ~SiPixelMonitorVertexSoA() override = default;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<ZVertexSoAHost> tokenSoAVertex_;
  edm::EDGetTokenT<reco::BeamSpot> tokenBeamSpot_;
  std::string topFolderName_;
  MonitorElement* hnVertex;
  MonitorElement* hx;
  MonitorElement* hy;
  MonitorElement* hz;
  MonitorElement* hchi2;
  MonitorElement* hchi2oNdof;
  MonitorElement* hptv2;
  MonitorElement* hntrks;
};

//
// constructors
//

SiPixelMonitorVertexSoA::SiPixelMonitorVertexSoA(const edm::ParameterSet& iConfig) {
  tokenSoAVertex_ = consumes(iConfig.getParameter<edm::InputTag>("pixelVertexSrc"));
  tokenBeamSpot_ = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotSrc"));
  topFolderName_ = iConfig.getParameter<std::string>("topFolderName");
}

//
// -- Analyze
//
void SiPixelMonitorVertexSoA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& vsoaHandle = iEvent.getHandle(tokenSoAVertex_);
  if (!vsoaHandle.isValid()) {
    edm::LogWarning("SiPixelMonitorVertexSoA") << "No Vertex SoA found \n returning!" << std::endl;
    return;
  }

  auto const& vsoa = *vsoaHandle;
  int nVertices = vsoa.view().nvFinal();
  auto bsHandle = iEvent.getHandle(tokenBeamSpot_);
  float x0 = 0., y0 = 0., z0 = 0., dxdz = 0., dydz = 0.;
  if (!bsHandle.isValid()) {
    edm::LogWarning("SiPixelMonitorVertexSoA") << "No beamspot found. returning vertexes with (0,0,Z) ";
  } else {
    const reco::BeamSpot& bs = *bsHandle;
    x0 = bs.x0();
    y0 = bs.y0();
    z0 = bs.z0();
    dxdz = bs.dxdz();
    dydz = bs.dydz();
  }

  for (int iv = 0; iv < nVertices; iv++) {
    auto si = vsoa.view()[iv].sortInd();
    auto z = vsoa.view()[si].zv();
    auto x = x0 + dxdz * z;
    auto y = y0 + dydz * z;

    z += z0;
    hx->Fill(x);
    hy->Fill(y);
    hz->Fill(z);
    auto ndof = vsoa.view()[si].ndof();
    hchi2->Fill(vsoa.view()[si].chi2());
    hchi2oNdof->Fill(vsoa.view()[si].chi2() / ndof);
    hptv2->Fill(vsoa.view()[si].ptv2());
    hntrks->Fill(ndof + 1);
  }
  hnVertex->Fill(nVertices);
}

//
// -- Book Histograms
//
void SiPixelMonitorVertexSoA::bookHistograms(DQMStore::IBooker& ibooker,
                                             edm::Run const& iRun,
                                             edm::EventSetup const& iSetup) {
  //std::string top_folder = ""//
  ibooker.cd();
  ibooker.setCurrentFolder(topFolderName_);
  hnVertex = ibooker.book1D("nVertex", ";# of Vertices;#entries", 101, -0.5, 100.5);
  hx = ibooker.book1D("vx", ";Vertex x;#entries", 10, -5., 5.);
  hy = ibooker.book1D("vy", ";Vertex y;#entries", 10, -5., 5.);
  hz = ibooker.book1D("vz", ";Vertex z;#entries", 30, -30., 30);
  hchi2 = ibooker.book1D("chi2", ";Vertex chi-squared;#entries", 40, 0., 20.);
  hchi2oNdof = ibooker.book1D("chi2oNdof", ";Vertex chi-squared/Ndof;#entries", 40, 0., 20.);
  hptv2 = ibooker.book1D("ptsq", ";Vertex #sum (p_{T})^{2};#entries", 200, 0., 200.);
  hntrks = ibooker.book1D("ntrk", ";#tracks associated;#entries", 100, -0.5, 99.5);
}

void SiPixelMonitorVertexSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelVertexSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelVertexSrc", edm::InputTag("pixelVerticesSoA"));
  desc.add<edm::InputTag>("beamSpotSrc", edm::InputTag("offlineBeamSpot"));
  desc.add<std::string>("topFolderName", "SiPixelHeterogeneous/PixelVertexSoA");
  descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(SiPixelMonitorVertexSoA);

using SiPixelPhase1MonitorVertexSoA = SiPixelMonitorVertexSoA;
DEFINE_FWK_MODULE(SiPixelPhase1MonitorVertexSoA);
