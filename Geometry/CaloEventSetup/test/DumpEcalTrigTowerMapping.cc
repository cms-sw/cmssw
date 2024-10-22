#include <memory>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

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

class DumpEcalTrigTowerMapping : public edm::one::EDAnalyzer<> {
public:
  explicit DumpEcalTrigTowerMapping(const edm::ParameterSet&);
  ~DumpEcalTrigTowerMapping() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  void build(const CaloGeometry& cg,
             const EcalTrigTowerConstituentsMap& etmap,
             DetId::Detector det,
             int subdetn,
             const char* name);
  int towerColor(const EcalTrigTowerDetId& theTower);

  edm::ESGetToken<EcalTrigTowerConstituentsMap, IdealGeometryRecord> eTTmapToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;

  int pass_;
};

DumpEcalTrigTowerMapping::DumpEcalTrigTowerMapping(const edm::ParameterSet& /*iConfig*/)
    : eTTmapToken_{esConsumes<EcalTrigTowerConstituentsMap, IdealGeometryRecord>(edm::ESInputTag{})},
      geometryToken_{esConsumes<CaloGeometry, CaloGeometryRecord>(edm::ESInputTag{})} {
  //now do what ever initialization is needed
  pass_ = 0;
  // some setup for root
  gROOT->SetStyle("Plain");  // white fill colors etc.
  gStyle->SetPaperSize(TStyle::kA4);
}

DumpEcalTrigTowerMapping::~DumpEcalTrigTowerMapping() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

int DumpEcalTrigTowerMapping::towerColor(const EcalTrigTowerDetId& theTower) {
  int iEtaColorIndex = (theTower.ietaAbs() - 1) % 2;
  int iPhiColorIndex = 0;
  if (theTower.ietaAbs() < 26)
    iPhiColorIndex = (theTower.iphi() - 1) % 2;
  else
    iPhiColorIndex = ((theTower.iphi() - 1) % 4) / 2;

  return iEtaColorIndex * 2 + iPhiColorIndex + 1;
}

void DumpEcalTrigTowerMapping::build(const CaloGeometry& cg,
                                     const EcalTrigTowerConstituentsMap& etmap,
                                     DetId::Detector det,
                                     int subdetn,
                                     const char* name) {
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
    const std::vector<DetId>& eeDetIds = cg.getValidDetIds(det, subdetn);

    edm::LogVerbatim("CaloGeom") << "*** testing endcap trig tower mapping **";
    for (const auto& eeDetId : eeDetIds) {
      EEDetId myId(eeDetId);
      EcalTrigTowerDetId myTower = etmap.towerOf(eeDetId);

      assert(myTower == EcalTrigTowerDetId::detIdFromDenseIndex(myTower.denseIndex()));

      if (myId.zside() == 1)
        continue;

      TBox* box = new TBox(myId.ix() - 0.5, myId.iy() - 0.5, myId.ix() + 0.5, myId.iy() + 0.5);
      box->SetFillColor(towerColor(myTower));
      box->Draw();
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
    const std::vector<DetId>& ebDetIds = cg.getValidDetIds(det, subdetn);

    edm::LogVerbatim("CaloGeom") << "*** testing barrel trig tower mapping **";
    for (const auto& ebDetId : ebDetIds) {
      EBDetId myId(ebDetId);
      EcalTrigTowerDetId myTower = etmap.towerOf(ebDetId);

      assert(myTower == EcalTrigTowerDetId::detIdFromDenseIndex(myTower.denseIndex()));

      TBox* box = new TBox(myId.ieta() - 0.5, myId.iphi() - 0.5, myId.ieta() + 0.5, myId.iphi() + 0.5);
      box->SetFillColor(towerColor(myTower));
      box->Draw();
    }
    gPad->SaveAs(name);
    delete canv;
    delete h;
  }
}

void DumpEcalTrigTowerMapping::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  edm::LogVerbatim("CaloGeom") << "Here I am ";

  const auto& eTTmap = iSetup.getData(eTTmapToken_);
  const auto& pG = iSetup.getData(geometryToken_);

  if (pass_ == 1) {
    build(pG, eTTmap, DetId::Ecal, EcalBarrel, "EBTTmapping.eps");
  }
  if (pass_ == 2) {
    build(pG, eTTmap, DetId::Ecal, EcalEndcap, "EETTmapping.eps");
  }

  pass_++;
}

//define this as a plug-in
DEFINE_FWK_MODULE(DumpEcalTrigTowerMapping);
