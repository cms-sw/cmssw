/*
 * \file DQMSourceEleCalib.cc
 *
 * \author Andrea Gozzelino - Universita� e INFN Torino
 * \author Stefano Argiro
 *
 *
 *
 * Description: Monitoring of Phi Symmetry Calibration Stream
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// DQM include files

#include "DQMServices/Core/interface/DQMStore.h"

// work on collections
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "DQMOffline/CalibCalo/interface/DQMSourceEleCalib.h"

using namespace std;
using namespace edm;

// ******************************************
// constructors
// *****************************************

DQMSourceEleCalib::DQMSourceEleCalib(const edm::ParameterSet &ps) : eventCounter_(0) {
  folderName_ = ps.getUntrackedParameter<string>("FolderName", "ALCAStreamEcalSingleEle");
  productMonitoredEB_ = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("AlCaStreamEBTag"));
  productMonitoredEE_ = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("AlCaStreamEETag"));

  saveToFile_ = ps.getUntrackedParameter<bool>("SaveToFile", false);
  fileName_ = ps.getUntrackedParameter<string>("FileName", "MonitorAlCaEcalSingleEle.root");
  productMonitoredElectrons_ = consumes<reco::GsfElectronCollection>(ps.getParameter<InputTag>("electronCollection"));
  prescaleFactor_ = ps.getUntrackedParameter<unsigned int>("prescaleFactor", 1);
}

DQMSourceEleCalib::~DQMSourceEleCalib() {}

//--------------------------------------------------------
void DQMSourceEleCalib::bookHistograms(DQMStore::IBooker &ibooker,
                                       edm::Run const &irun,
                                       edm::EventSetup const &isetup) {
  // create and cd into new folder
  ibooker.setCurrentFolder(folderName_);

  recHitsPerElectron_ = ibooker.book1D("recHitsPerElectron_", "recHitPerElectron", 200, 0, 200);
  ElectronsNumber_ = ibooker.book1D("ElectronsNumber_", "electrons in the event", 40, 0, 40);
  ESCoP_ = ibooker.book1D("ESCoP", "ESCoP", 50, 0, 5);

  OccupancyEB_ = ibooker.book2D("OccupancyEB_", "OccupancyEB", 360, 1, 361, 171, -85, 86);
  OccupancyEEP_ = ibooker.book2D("OccupancyEEP_", "Occupancy EE Plus", 100, 1, 101, 100, 1, 101);
  OccupancyEEM_ = ibooker.book2D("OccupancyEEM_", "Occupancy EE Minus", 100, 1, 101, 100, 1, 101);
  HitsVsAssociatedHits_ = ibooker.book1D("HitsVsAssociatedHits_", "HitsVsAssociatedHits", 100, 0, 5);
  LocalOccupancyEB_ = ibooker.book2D("LocalOccupancyEB_", "Local occupancy Barrel", 9, -4, 5, 9, -4, 5);
  LocalOccupancyEE_ = ibooker.book2D("LocalOccupancyEE_", "Local occupancy Endcap", 9, -4, 5, 9, -4, 5);
}

//--------------------------------------------------------

//-------------------------------------------------------------

void DQMSourceEleCalib::analyze(const Event &iEvent, const EventSetup &iSetup) {
  //  if (eventCounter_% prescaleFactor_ ) return; //FIXME
  eventCounter_++;
  int numberOfHits = 0;
  int numberOfElectrons = 0;
  int numberOfAssociatedHits = 0;
  // reads the recHits
  edm::Handle<EcalRecHitCollection> rhEB;
  edm::Handle<EcalRecHitCollection> rhEE;

  iEvent.getByToken(productMonitoredEB_, rhEB);
  iEvent.getByToken(productMonitoredEE_, rhEE);

  EcalRecHitCollection::const_iterator itb;

  // reads the electrons
  edm::Handle<reco::GsfElectronCollection> pElectrons;
  iEvent.getByToken(productMonitoredElectrons_, pElectrons);

  if (pElectrons.isValid()) {
    ElectronsNumber_->Fill(pElectrons->size() + 0.1);
    numberOfElectrons = pElectrons->size();
    for (reco::GsfElectronCollection::const_iterator eleIt = pElectrons->begin(); eleIt != pElectrons->end(); ++eleIt) {
      ESCoP_->Fill(eleIt->eSuperClusterOverP());
      numberOfAssociatedHits += eleIt->superCluster()->size();
      DetId Max = findMaxHit(eleIt->superCluster()->hitsAndFractions(), rhEB.product(), rhEE.product());
      if (!Max.det())
        continue;
      if (Max.subdetId() == EcalBarrel) {
        EBDetId EBMax(Max);
        fillAroundBarrel(rhEB.product(), EBMax.ieta(), EBMax.iphi());
      }
      if (Max.subdetId() == EcalEndcap) {
        EEDetId EEMax(Max);
        fillAroundEndcap(rhEE.product(), EEMax.ix(), EEMax.iy());
      }
    }
  }  // is valid electron

  // fill EB histos
  if (rhEB.isValid()) {
    numberOfHits += rhEB->size();
    for (itb = rhEB->begin(); itb != rhEB->end(); ++itb) {
      EBDetId id(itb->id());
      OccupancyEB_->Fill(id.iphi(), id.ieta());
    }  // Eb rechits
  }    // is Valid
  if (rhEE.isValid()) {
    numberOfHits += rhEE->size();
    for (itb = rhEE->begin(); itb != rhEE->end(); ++itb) {
      EEDetId id(itb->id());
      if (id.zside() > 0) {
        OccupancyEEP_->Fill(id.ix(), id.iy());
      }  // zside>0
      else if (id.zside() < 0) {
        OccupancyEEM_->Fill(id.ix(), id.iy());
      }  // zside<0

    }  // EE reChit
  }    // is Valid
  if (numberOfElectrons)
    recHitsPerElectron_->Fill((double)numberOfHits / ((double)numberOfElectrons));
  if (numberOfHits)
    HitsVsAssociatedHits_->Fill((double)numberOfAssociatedHits / ((double)numberOfHits));
}  // end of the analyzer

//--------------------------------------------------------

//------------------------------------------------

DetId DQMSourceEleCalib::findMaxHit(const std::vector<std::pair<DetId, float>> &v1,
                                    const EcalRecHitCollection *EBhits,
                                    const EcalRecHitCollection *EEhits) {
  double currEnergy = 0.;
  DetId maxHit;
  for (std::vector<std::pair<DetId, float>>::const_iterator idsIt = v1.begin(); idsIt != v1.end(); ++idsIt) {
    if (idsIt->first.subdetId() == EcalBarrel) {
      EcalRecHitCollection::const_iterator itrechit;
      itrechit = EBhits->find((*idsIt).first);
      if (itrechit == EBhits->end()) {
        edm::LogInfo("reading") << "[findMaxHit] rechit not found! ";
        continue;
      }
      // FIXME: wnat to use the fraction i.e. .second??
      if (itrechit->energy() > currEnergy) {
        currEnergy = itrechit->energy();
        maxHit = (*idsIt).first;
      }
    } else {
      EcalRecHitCollection::const_iterator itrechit;
      itrechit = EEhits->find((*idsIt).first);
      if (itrechit == EEhits->end()) {
        edm::LogInfo("reading") << "[findMaxHit] rechit not found! ";
        continue;
      }

      // FIXME: wnat to use the fraction i.e. .second??
      if (itrechit->energy() > currEnergy) {
        currEnergy = itrechit->energy();
        maxHit = (*idsIt).first;
      }
    }
  }
  return maxHit;
}

void DQMSourceEleCalib::fillAroundBarrel(const EcalRecHitCollection *recHits, int eta, int phi) {
  for (EcalRecHitCollection::const_iterator elem = recHits->begin(); elem != recHits->end(); ++elem) {
    EBDetId elementId = elem->id();
    LocalOccupancyEB_->Fill(elementId.ieta() - eta, elementId.iphi() - phi, elem->energy());
  }
  return;
}

// ----------------------------------------------------------------

void DQMSourceEleCalib::fillAroundEndcap(const EcalRecHitCollection *recHits, int ics, int ips) {
  for (EcalRecHitCollection::const_iterator elem = recHits->begin(); elem != recHits->end(); ++elem) {
    EEDetId elementId = elem->id();
    LocalOccupancyEE_->Fill(elementId.ix() - ics, elementId.iy() - ips, elem->energy());
  }
  return;
}
