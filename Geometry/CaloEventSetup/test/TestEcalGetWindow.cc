#include <memory>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include <TCanvas.h>
#include <TVirtualPad.h>
#include <TStyle.h>
#include <TROOT.h>
#include <TH2F.h>
#include <TBox.h>

#include <iostream>

class TestEcalGetWindow : public edm::one::EDAnalyzer<> {
public:
  explicit TestEcalGetWindow(const edm::ParameterSet&);
  ~TestEcalGetWindow() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  void build(const CaloGeometry& cg, const CaloTopology& etmap, DetId::Detector det, int subdetn, const char* name);
  int towerColor(const EcalTrigTowerDetId& theTower);

  edm::ESGetToken<CaloTopology, CaloTopologyRecord> topologyToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;

  int pass_;
};

TestEcalGetWindow::TestEcalGetWindow(const edm::ParameterSet& /*iConfig*/)
    : topologyToken_{esConsumes<CaloTopology, CaloTopologyRecord>(edm::ESInputTag{})},
      geometryToken_{esConsumes<CaloGeometry, CaloGeometryRecord>(edm::ESInputTag{})} {
  //now do what ever initialization is needed
  pass_ = 0;
  // some setup for root
  gROOT->SetStyle("Plain");  // white fill colors etc.
  gStyle->SetPaperSize(TStyle::kA4);
}

TestEcalGetWindow::~TestEcalGetWindow() {}

void TestEcalGetWindow::build(
    const CaloGeometry& /*cg*/, const CaloTopology& ct, DetId::Detector det, int subdetn, const char* name) {
  if (det == DetId::Ecal && subdetn == EcalEndcap) {
    TCanvas* canv = new TCanvas("c", "", 1000, 1000);
    canv->SetLeftMargin(0.15);
    canv->SetBottomMargin(0.15);

    gStyle->SetOptStat(0);
    TH2F* h = new TH2F("", "", 10, 0.5, 100.5, 10, 0.5, 100.5);

    h->Draw();
    //gPad->SetGridx();
    //gPad->SetGridy();
    gPad->Update();

    h->SetXTitle("x index");
    h->SetYTitle("y index");

    h->GetXaxis()->SetTickLength(-0.03);
    h->GetYaxis()->SetTickLength(-0.03);

    h->GetXaxis()->SetLabelOffset(0.03);
    h->GetYaxis()->SetLabelOffset(0.03);

    h->GetXaxis()->SetLabelSize(0.04);
    h->GetYaxis()->SetLabelSize(0.04);

    // axis titles
    h->GetXaxis()->SetTitleSize(0.04);
    h->GetYaxis()->SetTitleSize(0.04);

    h->GetXaxis()->SetTitleOffset(1.8);
    h->GetYaxis()->SetTitleOffset(1.9);

    h->GetXaxis()->CenterTitle(true);
    h->GetYaxis()->CenterTitle(true);
    const CaloSubdetectorTopology* topology = ct.getSubdetectorTopology(det, subdetn);

    std::vector<DetId> eeDetIds;
    eeDetIds.emplace_back(EEDetId(1, 50, 1, EEDetId::XYMODE));
    eeDetIds.emplace_back(EEDetId(25, 50, 1, EEDetId::XYMODE));
    eeDetIds.emplace_back(EEDetId(50, 1, 1, EEDetId::XYMODE));
    eeDetIds.emplace_back(EEDetId(50, 25, 1, EEDetId::XYMODE));
    eeDetIds.emplace_back(EEDetId(3, 60, 1, EEDetId::XYMODE));
    for (const auto& eeDetId : eeDetIds) {
      EEDetId myId(eeDetId);
      if (myId.zside() == -1)
        continue;
      std::vector<DetId> myNeighbours = topology->getWindow(myId, 13, 13);
      for (const auto& myNeighbour : myNeighbours) {
        EEDetId myEEId(myNeighbour);
        TBox* box = new TBox(myEEId.ix() - 0.5, myEEId.iy() - 0.5, myEEId.ix() + 0.5, myEEId.iy() + 0.5);
        box->SetFillColor(1);
        box->Draw();
      }
    }
    gPad->SaveAs(name);
    delete canv;
    delete h;
  }

  if (det == DetId::Ecal && subdetn == EcalBarrel) {
    TCanvas* canv = new TCanvas("c", "", 1000, 1000);
    canv->SetLeftMargin(0.15);
    canv->SetBottomMargin(0.15);

    gStyle->SetOptStat(0);
    TH2F* h = new TH2F("", "", 10, -85.5, 85.5, 10, 0.5, 360.5);

    h->Draw();
    //gPad->SetGridx();
    //gPad->SetGridy();
    gPad->Update();

    h->SetXTitle("eta index");
    h->SetYTitle("phi index");

    h->GetXaxis()->SetTickLength(-0.03);
    h->GetYaxis()->SetTickLength(-0.03);

    h->GetXaxis()->SetLabelOffset(0.03);
    h->GetYaxis()->SetLabelOffset(0.03);

    h->GetXaxis()->SetLabelSize(0.04);
    h->GetYaxis()->SetLabelSize(0.04);

    // axis titles
    h->GetXaxis()->SetTitleSize(0.04);
    h->GetYaxis()->SetTitleSize(0.04);

    h->GetXaxis()->SetTitleOffset(1.8);
    h->GetYaxis()->SetTitleOffset(1.9);

    h->GetXaxis()->CenterTitle(true);
    h->GetYaxis()->CenterTitle(true);
    const CaloSubdetectorTopology* topology = ct.getSubdetectorTopology(det, subdetn);
    std::vector<DetId> ebDetIds;
    ebDetIds.emplace_back(EBDetId(1, 1));
    ebDetIds.emplace_back(EBDetId(30, 30));
    ebDetIds.emplace_back(EBDetId(-1, 120));
    ebDetIds.emplace_back(EBDetId(85, 1));
    for (const auto& ebDetId : ebDetIds) {
      EBDetId myId(ebDetId);
      std::vector<DetId> myNeighbours = topology->getWindow(myId, 13, 13);
      for (const auto& myNeighbour : myNeighbours) {
        EBDetId myEBId(myNeighbour);
        TBox* box = new TBox(myEBId.ieta() - 0.5, myEBId.iphi() - 0.5, myEBId.ieta() + 0.5, myEBId.iphi() + 0.5);
        box->SetFillColor(1);
        box->Draw();
      }
    }
    gPad->SaveAs(name);
    delete canv;
    delete h;
  }
}
// ------------ method called to produce the data  ------------
void TestEcalGetWindow::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  edm::LogVerbatim("CaloGeom") << "Here I am ";

  const auto& theCaloTopology = iSetup.getData(topologyToken_);
  const auto& pG = iSetup.getData(geometryToken_);

  if (pass_ == 1) {
    build(pG, theCaloTopology, DetId::Ecal, EcalBarrel, "EBGetWindowTest.eps");
  }
  if (pass_ == 2) {
    build(pG, theCaloTopology, DetId::Ecal, EcalEndcap, "EEGetWindowTest.eps");
  }

  pass_++;
}

//define this as a plug-in

DEFINE_FWK_MODULE(TestEcalGetWindow);
