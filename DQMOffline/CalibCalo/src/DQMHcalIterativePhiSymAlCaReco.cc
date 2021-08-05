/*
 * \file DQMHcalIterativePhiSymAlCaReco.cc
 *
 * \author Sunanda Banerjee
 *
 *
 *
 * Description: Monitoring of Iterative Phi Symmetry Calibration Stream
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// DQM include files

#include "DQMServices/Core/interface/DQMStore.h"

// work on collections

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DQMOffline/CalibCalo/interface/DQMHcalIterativePhiSymAlCaReco.h"

// ******************************************
// constructors
// *****************************************

DQMHcalIterativePhiSymAlCaReco::DQMHcalIterativePhiSymAlCaReco(const edm::ParameterSet &ps) {
  //
  // Input from configurator file
  //
  folderName_ = ps.getParameter<std::string>("folderName");

  // histogram parameters
  tok_ho_ = consumes<HORecHitCollection>(ps.getParameter<edm::InputTag>("hoInput"));
  tok_hf_ = consumes<HFRecHitCollection>(ps.getParameter<edm::InputTag>("hfInput"));
  tok_hbhe_ = consumes<HBHERecHitCollection>(ps.getParameter<edm::InputTag>("hbheInput"));

  // Distribution of rechits in iPhi, iEta
  hiDistr_y_nbin_ = ps.getUntrackedParameter<int>("hiDistr_y_nbin", 72);
  hiDistr_y_min_ = ps.getUntrackedParameter<double>("hiDistr_y_min", 0.5);
  hiDistr_y_max_ = ps.getUntrackedParameter<double>("hiDistr_y_max", 72.5);
  hiDistr_x_nbin_ = ps.getUntrackedParameter<int>("hiDistr_x_nbin", 83);
  hiDistr_x_min_ = ps.getUntrackedParameter<double>("hiDistr_x_min", -41.5);
  hiDistr_x_max_ = ps.getUntrackedParameter<double>("hiDistr_x_max", 41.5);
}

DQMHcalIterativePhiSymAlCaReco::~DQMHcalIterativePhiSymAlCaReco() {}

void DQMHcalIterativePhiSymAlCaReco::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folderName", "ALCAStreamHcalIterativePhiSym");
  desc.add<edm::InputTag>("hbheInput", edm::InputTag("hbhereco"));
  desc.add<edm::InputTag>("hfInput", edm::InputTag("hfreco"));
  desc.add<edm::InputTag>("hoInput", edm::InputTag("horeco"));
  desc.addUntracked<int>("hiDistr_y_nbin", 72);
  desc.addUntracked<double>("hiDistr_y_min", 0.5);
  desc.addUntracked<double>("hiDistr_y_max", 72.5);
  desc.addUntracked<int>("hiDistr_x_nbin", 83);
  desc.addUntracked<double>("hiDistr_x_min", -41.5);
  desc.addUntracked<double>("hiDistr_x_max", 41.5);
  descriptions.add("dqmHcalIterativePhiSymAlCaReco", desc);
}

//--------------------------------------------------------
void DQMHcalIterativePhiSymAlCaReco::bookHistograms(DQMStore::IBooker &ibooker,
                                                    edm::Run const &irun,
                                                    edm::EventSetup const &isetup) {
  // create and cd into new folder
  ibooker.setCurrentFolder(folderName_);

  // book some histograms 1D
  hiDistrHBHEsize1D_ = ibooker.book1D("DistrHBHEsize", "Size of HBHE Collection", 100, 0.0, 10000.0);
  hiDistrHFsize1D_ = ibooker.book1D("DistrHFsize", "Size of HF Collection", 100, 0.0, 10000.0);
  hiDistrHOsize1D_ = ibooker.book1D("DistrHOsize", "Size of HO Collection", 100, 0.0, 3000.0);

  // Eta-phi occupancy
  for (int k = 0; k < maxDepth_; ++k) {
    char name[20], title[20];
    sprintf(name, "MBdepth%d", (k + 1));
    sprintf(title, "Depth %d", (k + 1));
    hiDistr2D_[k] = ibooker.book2D(
        name, title, hiDistr_x_nbin_, hiDistr_x_min_, hiDistr_x_max_, hiDistr_y_nbin_, hiDistr_y_min_, hiDistr_y_max_);
    hiDistr2D_[k]->setAxisTitle("i#phi ", 2);
    hiDistr2D_[k]->setAxisTitle("i#eta ", 1);
  }
}

//--------------------------------------------------------

//-------------------------------------------------------------

void DQMHcalIterativePhiSymAlCaReco::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // First HBHE RecHits
  edm::Handle<HBHERecHitCollection> hbheMB;
  iEvent.getByToken(tok_hbhe_, hbheMB);
  if (!hbheMB.isValid()) {
    edm::LogWarning("DQMHcal") << "DQMHcalIterativePhiSymAlCaReco: Error! can't get HBHE RecHit Collection!";
  } else {
    const HBHERecHitCollection Hithbhe = *(hbheMB.product());
    hiDistrHBHEsize1D_->Fill(Hithbhe.size());
    for (HBHERecHitCollection::const_iterator hbheItr = Hithbhe.begin(); hbheItr != Hithbhe.end(); hbheItr++) {
      HcalDetId hid = HcalDetId(hbheItr->detid());
      int id = hid.depth() - 1;
      if ((id >= 0) && (id < maxDepth_))
        hiDistr2D_[id]->Fill(hid.ieta(), hid.iphi(), hbheItr->energy());
    }
  }

  // Then HF RecHits
  edm::Handle<HFRecHitCollection> hfMB;
  iEvent.getByToken(tok_hf_, hfMB);
  if (!hfMB.isValid()) {
    edm::LogWarning("DQMHcal") << "DQMHcalIterativePhiSymAlCaReco: Error! can't get HF RecHit Collection!";
  } else {
    const HFRecHitCollection Hithf = *(hfMB.product());
    hiDistrHFsize1D_->Fill(Hithf.size());
    for (HFRecHitCollection::const_iterator hfItr = Hithf.begin(); hfItr != Hithf.end(); hfItr++) {
      HcalDetId hid = HcalDetId(hfItr->detid());
      int id = hid.depth() - 1;
      if ((id >= 0) && (id < maxDepth_))
        hiDistr2D_[id]->Fill(hid.ieta(), hid.iphi(), hfItr->energy());
    }
  }

  // And finally HO RecHits
  edm::Handle<HORecHitCollection> hoMB;
  iEvent.getByToken(tok_ho_, hoMB);
  if (!hoMB.isValid()) {
    edm::LogWarning("DQMHcal") << "DQMHcalIterativePhiSymAlCaReco: Error! can't get HO RecHit Collection!";
  } else {
    const HORecHitCollection Hitho = *(hoMB.product());
    hiDistrHOsize1D_->Fill(Hitho.size());
    for (HORecHitCollection::const_iterator hoItr = Hitho.begin(); hoItr != Hitho.end(); hoItr++) {
      HcalDetId hid = HcalDetId(hoItr->detid());
      int id = hid.depth() - 1;
      if ((id >= 0) && (id < maxDepth_))
        hiDistr2D_[id]->Fill(hid.ieta(), hid.iphi(), hoItr->energy());
    }
  }

}  // analyze
