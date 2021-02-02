#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include <TH1F.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class ElectronStudy : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit ElectronStudy(const edm::ParameterSet& ps);
  ~ElectronStudy() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override {}

  static const int NEtaBins_ = 3;
  static const int NPBins_ = 8;
  double pBins_[NPBins_ + 1], etaBins_[NEtaBins_ + 1];

  edm::EDGetTokenT<edm::PCaloHitContainer> tok_EBhit_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_EEhit_;
  edm::EDGetTokenT<edm::SimTrackContainer> tok_simTk_;
  edm::EDGetTokenT<edm::SimVertexContainer> tok_simVtx_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  edm::ESGetToken<CaloTopology, CaloTopologyRecord> tok_caloTopology_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_magField_;

  int hotZone_, verbose_;
  bool histos_;
  TH1F* histoR1[NPBins_ + 1][NEtaBins_ + 1];
  TH1F* histoR2[NPBins_ + 1][NEtaBins_ + 1];
  TH1F* histoR3[NPBins_ + 1][NEtaBins_ + 1];
  TH1F* histoE1x1[NPBins_ + 1][NEtaBins_ + 1];
  TH1F* histoE3x3[NPBins_ + 1][NEtaBins_ + 1];
  TH1F* histoE5x5[NPBins_ + 1][NEtaBins_ + 1];
};

ElectronStudy::ElectronStudy(const edm::ParameterSet& ps) {
  usesResource("TFileService");

  std::string g4Label = ps.getUntrackedParameter<std::string>("ModuleLabel", "g4SimHits");
  std::string hitLabEB = ps.getUntrackedParameter<std::string>("EBCollection", "EcalHitsEB");
  std::string hitLabEE = ps.getUntrackedParameter<std::string>("EECollection", "EcalHitsEE");

  tok_EBhit_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label, hitLabEB));
  tok_EEhit_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label, hitLabEE));
  tok_simTk_ = consumes<edm::SimTrackContainer>(edm::InputTag(g4Label));
  tok_simVtx_ = consumes<edm::SimVertexContainer>(edm::InputTag(g4Label));

  hotZone_ = ps.getUntrackedParameter<int>("HotZone", 0);
  verbose_ = ps.getUntrackedParameter<int>("Verbosity", 0);
  edm::LogInfo("ElectronStudy") << "Module Label: " << g4Label << "   Hits: " << hitLabEB << ", " << hitLabEE;

  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  tok_caloTopology_ = esConsumes<CaloTopology, CaloTopologyRecord>();
  tok_magField_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();

  double tempP[NPBins_ + 1] = {0.0, 10.0, 20.0, 40.0, 60.0, 100.0, 500.0, 1000.0, 10000.0};
  double tempEta[NEtaBins_ + 1] = {0.0, 1.2, 1.6, 3.0};

  for (int i = 0; i < NPBins_ + 1; i++)
    pBins_[i] = tempP[i];
  for (int i = 0; i < NEtaBins_ + 1; i++)
    etaBins_[i] = tempEta[i];

  edm::Service<TFileService> tfile;
  if (!tfile.isAvailable()) {
    edm::LogInfo("ElectronStudy") << "TFileService unavailable: no histograms";
    histos_ = false;
  } else {
    char name[20], title[200], cpbin[30], cebin[30];
    histos_ = true;
    for (unsigned int i = 0; i < NPBins_ + 1; ++i) {
      if (i == 0)
        sprintf(cpbin, " All p");
      else
        sprintf(cpbin, " p (%6.0f:%6.0f)", pBins_[i - 1], pBins_[i]);
      for (unsigned int j = 0; j < NEtaBins_ + 1; ++j) {
        if (j == 0)
          sprintf(cebin, " All #eta");
        else
          sprintf(cebin, " #eta (%4.1f:%4.1f)", etaBins_[j - 1], etaBins_[j]);
        sprintf(name, "R1%d%d", i, j);
        sprintf(title, "E1/E9 for %s%s", cpbin, cebin);
        histoR1[i][j] = tfile->make<TH1F>(name, title, 100, 0., 2.);
        histoR1[i][j]->GetXaxis()->SetTitle(title);
        histoR1[i][j]->GetYaxis()->SetTitle("Tracks");
        sprintf(name, "R2%d%d", i, j);
        sprintf(title, "E1/E25 for %s%s", cpbin, cebin);
        histoR2[i][j] = tfile->make<TH1F>(name, title, 100, 0., 2.);
        histoR2[i][j]->GetXaxis()->SetTitle(title);
        histoR2[i][j]->GetYaxis()->SetTitle("Tracks");
        sprintf(name, "R3%d%d", i, j);
        sprintf(title, "E9/E25 for %s%s", cpbin, cebin);
        histoR3[i][j] = tfile->make<TH1F>(name, title, 100, 0., 2.);
        histoR3[i][j]->GetXaxis()->SetTitle(title);
        histoR3[i][j]->GetYaxis()->SetTitle("Tracks");
        sprintf(name, "E1x1%d%d", i, j);
        sprintf(title, "E1/P for %s%s", cpbin, cebin);
        histoE1x1[i][j] = tfile->make<TH1F>(name, title, 100, 0., 2.);
        histoE1x1[i][j]->GetXaxis()->SetTitle(title);
        histoE1x1[i][j]->GetYaxis()->SetTitle("Tracks");
        sprintf(name, "E3x3%d%d", i, j);
        sprintf(title, "E9/P for %s%s", cpbin, cebin);
        histoE3x3[i][j] = tfile->make<TH1F>(name, title, 100, 0., 2.);
        histoE3x3[i][j]->GetXaxis()->SetTitle(title);
        histoE3x3[i][j]->GetYaxis()->SetTitle("Tracks");
        sprintf(name, "E5x5%d%d", i, j);
        sprintf(title, "E25/P for %s%s", cpbin, cebin);
        histoE5x5[i][j] = tfile->make<TH1F>(name, title, 100, 0., 2.);
        histoE5x5[i][j]->GetXaxis()->SetTitle(title);
        histoE5x5[i][j]->GetYaxis()->SetTitle("Tracks");
      }
    }
  }
}

void ElectronStudy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("ModuleLabel", "g4SimHits");
  desc.addUntracked<std::string>("EBCollection", "EcalHitsEB");
  desc.addUntracked<std::string>("EECollection", "EcalHitsEE");
  desc.addUntracked<int>("Verbosity", 0);
  descriptions.add("electronStudy", desc);
}

void ElectronStudy::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (verbose_ > 1)
    edm::LogVerbatim("IsoTrack") << "Run = " << iEvent.id().run() << " Event = " << iEvent.id().event();

  // get Geometry, B-field, Topology
  const CaloGeometry* geo = &iSetup.getData(tok_geom_);
  const MagneticField* bField = &iSetup.getData(tok_magField_);
  const CaloTopology* caloTopology = &iSetup.getData(tok_caloTopology_);

  // get PCaloHits for ecal barrel
  edm::Handle<edm::PCaloHitContainer> caloHitEB;
  iEvent.getByToken(tok_EBhit_, caloHitEB);

  // get PCaloHits for ecal endcap
  edm::Handle<edm::PCaloHitContainer> caloHitEE;
  iEvent.getByToken(tok_EEhit_, caloHitEE);

  // get sim tracks
  edm::Handle<edm::SimTrackContainer> SimTk;
  iEvent.getByToken(tok_simTk_, SimTk);

  // get sim vertices
  edm::Handle<edm::SimVertexContainer> SimVtx;
  iEvent.getByToken(tok_simVtx_, SimVtx);

  if (verbose_ > 0)
    edm::LogVerbatim("IsoTrack") << "ElectronStudy: hits valid[EB]: " << caloHitEB.isValid()
                                 << " valid[EE]: " << caloHitEE.isValid();

  if (caloHitEB.isValid() && caloHitEE.isValid()) {
    unsigned int indx;
    if (verbose_ > 2) {
      edm::PCaloHitContainer::const_iterator ihit;
      for (ihit = caloHitEB->begin(), indx = 0; ihit != caloHitEB->end(); ihit++, indx++) {
        EBDetId id = ihit->id();
        edm::LogVerbatim("IsoTrack") << "Hit[" << indx << "] " << id << " E " << ihit->energy() << " T "
                                     << ihit->time();
      }
      for (ihit = caloHitEE->begin(), indx = 0; ihit != caloHitEE->end(); ihit++, indx++) {
        EEDetId id = ihit->id();
        edm::LogVerbatim("IsoTrack") << "Hit[" << indx << "] " << id << " E " << ihit->energy() << " T "
                                     << ihit->time();
      }
    }
    edm::SimTrackContainer::const_iterator simTrkItr = SimTk->begin();
    for (indx = 0; simTrkItr != SimTk->end(); simTrkItr++, indx++) {
      if (verbose_ > 0)
        edm::LogVerbatim("IsoTrack") << "ElectronStudy: Track[" << indx << "] ID " << simTrkItr->trackId() << " type "
                                     << simTrkItr->type() << " charge " << simTrkItr->charge() << " p "
                                     << simTrkItr->momentum() << " Generator Index " << simTrkItr->genpartIndex()
                                     << " vertex " << simTrkItr->vertIndex();
      if (std::abs(simTrkItr->type()) == 11 && simTrkItr->vertIndex() != -1) {
        int thisTrk = simTrkItr->trackId();
        spr::propagatedTrackDirection trkD = spr::propagateCALO(thisTrk, SimTk, SimVtx, geo, bField, (verbose_ > 1));
        if (trkD.okECAL) {
          const DetId isoCell = trkD.detIdECAL;
          DetId hotCell = isoCell;
          if (hotZone_ > 0)
            hotCell = spr::hotCrystal(
                isoCell, caloHitEB, caloHitEE, geo, caloTopology, hotZone_, hotZone_, -500.0, 500.0, (verbose_ > 1));
          double e1x1 = spr::eECALmatrix(
              hotCell, caloHitEB, caloHitEE, geo, caloTopology, 0, 0, -100.0, -100.0, -500.0, 500.0, (verbose_ > 2));
          double e3x3 = spr::eECALmatrix(
              hotCell, caloHitEB, caloHitEE, geo, caloTopology, 1, 1, -100.0, -100.0, -500.0, 500.0, (verbose_ > 2));
          double e5x5 = spr::eECALmatrix(
              hotCell, caloHitEB, caloHitEE, geo, caloTopology, 2, 2, -100.0, -100.0, -500.0, 500.0, (verbose_ > 2));
          double p = simTrkItr->momentum().P();
          double eta = std::abs(simTrkItr->momentum().eta());
          int etaBin = -1, momBin = -1;
          for (int ieta = 0; ieta < NEtaBins_; ieta++) {
            if (eta > etaBins_[ieta] && eta < etaBins_[ieta + 1])
              etaBin = ieta + 1;
          }
          for (int ipt = 0; ipt < NPBins_; ipt++) {
            if (p > pBins_[ipt] && p < pBins_[ipt + 1])
              momBin = ipt + 1;
          }
          double r1 = -1, r2 = -1, r3 = -1;
          if (e3x3 > 0)
            r1 = e1x1 / e3x3;
          if (e5x5 > 0) {
            r2 = e1x1 / e5x5;
            r3 = e3x3 / e5x5;
          }
          if (verbose_ > 0) {
            edm::LogVerbatim("IsoTrack") << "ElectronStudy: p " << p << " [" << momBin << "] eta " << eta << " ["
                                         << etaBin << "] Cell 0x" << std::hex << isoCell() << std::dec;
            if (isoCell.subdetId() == EcalBarrel) {
              edm::LogVerbatim("IsoTrack") << EBDetId(isoCell);
            } else if (isoCell.subdetId() == EcalEndcap) {
              edm::LogVerbatim("IsoTrack") << EEDetId(isoCell);
            }
            edm::LogVerbatim("IsoTrack") << " e1x1 " << e1x1 << "|" << r1 << "|" << r2 << " e3x3 " << e3x3 << "|" << r3
                                         << " e5x5 " << e5x5;
          }
          if (histos_) {
            histoR1[0][0]->Fill(r1);
            histoR2[0][0]->Fill(r2);
            histoR3[0][0]->Fill(r3);
            histoE1x1[0][0]->Fill(e1x1 / p);
            histoE3x3[0][0]->Fill(e3x3 / p);
            histoE5x5[0][0]->Fill(e5x5 / p);
            if (momBin > 0) {
              histoR1[momBin][0]->Fill(r1);
              histoR2[momBin][0]->Fill(r2);
              histoR3[momBin][0]->Fill(r3);
              histoE1x1[momBin][0]->Fill(e1x1 / p);
              histoE3x3[momBin][0]->Fill(e3x3 / p);
              histoE5x5[momBin][0]->Fill(e5x5 / p);
            }
            if (etaBin > 0) {
              histoR1[0][etaBin]->Fill(r1);
              histoR2[0][etaBin]->Fill(r2);
              histoR3[0][etaBin]->Fill(r3);
              histoE1x1[0][etaBin]->Fill(e1x1 / p);
              histoE3x3[0][etaBin]->Fill(e3x3 / p);
              histoE5x5[0][etaBin]->Fill(e5x5 / p);
              if (momBin > 0) {
                histoR1[momBin][etaBin]->Fill(r1);
                histoR2[momBin][etaBin]->Fill(r2);
                histoR3[momBin][etaBin]->Fill(r3);
                histoE1x1[momBin][etaBin]->Fill(e1x1 / p);
                histoE3x3[momBin][etaBin]->Fill(e3x3 / p);
                histoE5x5[momBin][etaBin]->Fill(e5x5 / p);
              }
            }
          }
        }
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronStudy);
