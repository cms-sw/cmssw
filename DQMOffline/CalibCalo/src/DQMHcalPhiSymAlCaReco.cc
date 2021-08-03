/*
 * \file DQMHcalPhiSymAlCaReco.cc
 *
 * \author Olga Kodolova
 *
 *
 *
 * Description: Monitoring of Phi Symmetry Calibration Stream
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
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"

#include "DQMOffline/CalibCalo/src/DQMHcalPhiSymAlCaReco.h"

using namespace std;
using namespace edm;

// ******************************************
// constructors
// *****************************************

DQMHcalPhiSymAlCaReco::DQMHcalPhiSymAlCaReco(const edm::ParameterSet &ps) : eventCounter_(0) {
  //
  // Input from configurator file
  //
  folderName_ = ps.getUntrackedParameter<string>("FolderName", "ALCAStreamHcalPhiSym");

  hbherecoMB = consumes<HBHERecHitCollection>(ps.getParameter<edm::InputTag>("hbheInputMB"));
  horecoMB = ps.getParameter<edm::InputTag>("hoInputMB");
  hfrecoMB = consumes<HFRecHitCollection>(ps.getParameter<edm::InputTag>("hfInputMB"));

  hbherecoNoise = consumes<HBHERecHitCollection>(ps.getParameter<edm::InputTag>("hbheInputNoise"));
  horecoNoise = ps.getParameter<edm::InputTag>("hoInputNoise");
  hfrecoNoise = consumes<HFRecHitCollection>(ps.getParameter<edm::InputTag>("hfInputNoise"));

  rawInLabel_ = consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("rawInputLabel"));

  period_ = ps.getParameter<unsigned int>("period");

  saveToFile_ = ps.getUntrackedParameter<bool>("SaveToFile", false);
  fileName_ = ps.getUntrackedParameter<string>("FileName", "MonitorAlCaHcalPhiSym.root");

  perLSsaving_ = (ps.getUntrackedParameter<bool>("perLSsaving", false));

  // histogram parameters

  // Distribution of rechits in iPhi, iEta
  hiDistr_y_nbin_ = ps.getUntrackedParameter<int>("hiDistr_y_nbin", 72);
  hiDistr_y_min_ = ps.getUntrackedParameter<double>("hiDistr_y_min", 0.5);
  hiDistr_y_max_ = ps.getUntrackedParameter<double>("hiDistr_y_max", 72.5);
  hiDistr_x_nbin_ = ps.getUntrackedParameter<int>("hiDistr_x_nbin", 41);
  hiDistr_x_min_ = ps.getUntrackedParameter<double>("hiDistr_x_min", 0.5);
  hiDistr_x_max_ = ps.getUntrackedParameter<double>("hiDistr_x_max", 41.5);
  // Check for NZS
  hiDistr_r_nbin_ = ps.getUntrackedParameter<int>("hiDistr_r_nbin", 100);
  ihbhe_size_ = ps.getUntrackedParameter<double>("ihbhe_size_", 5184.);
  ihf_size_ = ps.getUntrackedParameter<double>("ihf_size_", 1728.);
}

DQMHcalPhiSymAlCaReco::~DQMHcalPhiSymAlCaReco() {}

//--------------------------------------------------------
void DQMHcalPhiSymAlCaReco::bookHistograms(DQMStore::IBooker &ibooker,
                                           edm::Run const &irun,
                                           edm::EventSetup const &isetup) {
  // create and cd into new folder
  ibooker.setCurrentFolder(folderName_);

  eventCounter_ = 0;

  hFEDsize = ibooker.book1D("hFEDsize", "HCAL FED size (kB)", 200, -0.5, 20.5);
  hFEDsize->setAxisTitle("kB", 1);

  hHcalIsZS = ibooker.book1D("hHcalIsZS", "Hcal Is ZS", 4, -1.5, 2.5);
  hHcalIsZS->setBinLabel(2, "NZS");
  hHcalIsZS->setBinLabel(3, "ZS");

  char hname[50];
  sprintf(hname, "L1 Event Number %% %i", period_);
  hL1Id = ibooker.book1D("hL1Id", hname, 4200, -99.5, 4099.5);
  hL1Id->setAxisTitle(hname);

  // book some histograms 1D
  double xmin = 0.1;
  double xmax = 1.1;
  hiDistrHBHEsize1D_ = ibooker.book1D("DistrHBHEsize", "Size of HBHE Collection", hiDistr_r_nbin_, xmin, xmax);
  hiDistrHFsize1D_ = ibooker.book1D("DistrHFsize", "Size of HF Collection", hiDistr_r_nbin_, xmin, xmax);

  // First moment
  hiDistrMBPl2D_ = ibooker.book2D("MBdepthPl1",
                                  "iphi- +ieta signal distribution at depth1",
                                  hiDistr_x_nbin_,
                                  hiDistr_x_min_,
                                  hiDistr_x_max_,
                                  hiDistr_y_nbin_,
                                  hiDistr_y_min_,
                                  hiDistr_y_max_);

  hiDistrMBPl2D_->setAxisTitle("i#phi ", 2);
  hiDistrMBPl2D_->setAxisTitle("i#eta ", 1);

  hiDistrNoisePl2D_ = ibooker.book2D("NoisedepthPl1",
                                     "iphi-ieta noise distribution at depth1",
                                     hiDistr_x_nbin_ + 1,
                                     hiDistr_x_min_ - 1.,
                                     hiDistr_x_max_,
                                     hiDistr_y_nbin_ + 1,
                                     hiDistr_y_min_ - 1.,
                                     hiDistr_y_max_);

  hiDistrNoisePl2D_->setAxisTitle("i#phi ", 2);
  hiDistrNoisePl2D_->setAxisTitle("i#eta ", 1);
  // Second moment
  hiDistrMB2Pl2D_ = ibooker.book2D("MB2depthPl1",
                                   "iphi- +ieta signal distribution at depth1",
                                   hiDistr_x_nbin_,
                                   hiDistr_x_min_,
                                   hiDistr_x_max_,
                                   hiDistr_y_nbin_,
                                   hiDistr_y_min_,
                                   hiDistr_y_max_);

  hiDistrMB2Pl2D_->setAxisTitle("i#phi ", 2);
  hiDistrMB2Pl2D_->setAxisTitle("i#eta ", 1);

  hiDistrNoise2Pl2D_ = ibooker.book2D("Noise2depthPl1",
                                      "iphi-ieta noise distribution at depth1",
                                      hiDistr_x_nbin_,
                                      hiDistr_x_min_,
                                      hiDistr_x_max_,
                                      hiDistr_y_nbin_,
                                      hiDistr_y_min_,
                                      hiDistr_y_max_);

  hiDistrNoise2Pl2D_->setAxisTitle("i#phi ", 2);
  hiDistrNoise2Pl2D_->setAxisTitle("i#eta ", 1);

  // Variance
  hiDistrVarMBPl2D_ = ibooker.book2D("VarMBdepthPl1",
                                     "iphi- +ieta signal distribution at depth1",
                                     hiDistr_x_nbin_,
                                     hiDistr_x_min_,
                                     hiDistr_x_max_,
                                     hiDistr_y_nbin_,
                                     hiDistr_y_min_,
                                     hiDistr_y_max_);

  hiDistrVarMBPl2D_->setAxisTitle("i#phi ", 2);
  hiDistrVarMBPl2D_->setAxisTitle("i#eta ", 1);

  hiDistrVarNoisePl2D_ = ibooker.book2D("VarNoisedepthPl1",
                                        "iphi-ieta noise distribution at depth1",
                                        hiDistr_x_nbin_,
                                        hiDistr_x_min_,
                                        hiDistr_x_max_,
                                        hiDistr_y_nbin_,
                                        hiDistr_y_min_,
                                        hiDistr_y_max_);

  hiDistrVarNoisePl2D_->setAxisTitle("i#phi ", 2);
  hiDistrVarNoisePl2D_->setAxisTitle("i#eta ", 1);

  //==================================================================================
  // First moment
  hiDistrMBMin2D_ = ibooker.book2D("MBdepthMin1",
                                   "iphi- +ieta signal distribution at depth1",
                                   hiDistr_x_nbin_,
                                   hiDistr_x_min_,
                                   hiDistr_x_max_,
                                   hiDistr_y_nbin_,
                                   hiDistr_y_min_,
                                   hiDistr_y_max_);

  hiDistrMBMin2D_->setAxisTitle("i#phi ", 2);
  hiDistrMBMin2D_->setAxisTitle("i#eta ", 1);

  hiDistrNoiseMin2D_ = ibooker.book2D("NoisedepthMin1",
                                      "iphi-ieta noise distribution at depth1",
                                      hiDistr_x_nbin_,
                                      hiDistr_x_min_,
                                      hiDistr_x_max_,
                                      hiDistr_y_nbin_,
                                      hiDistr_y_min_,
                                      hiDistr_y_max_);

  hiDistrNoiseMin2D_->setAxisTitle("i#phi ", 2);
  hiDistrNoiseMin2D_->setAxisTitle("i#eta ", 1);
  // Second moment
  hiDistrMB2Min2D_ = ibooker.book2D("MB2depthMin1",
                                    "iphi- +ieta signal distribution at depth1",
                                    hiDistr_x_nbin_,
                                    hiDistr_x_min_,
                                    hiDistr_x_max_,
                                    hiDistr_y_nbin_,
                                    hiDistr_y_min_,
                                    hiDistr_y_max_);

  hiDistrMB2Min2D_->setAxisTitle("i#phi ", 2);
  hiDistrMB2Min2D_->setAxisTitle("i#eta ", 1);

  hiDistrNoise2Min2D_ = ibooker.book2D("Noise2depthMin1",
                                       "iphi-ieta noise distribution at depth1",
                                       hiDistr_x_nbin_,
                                       hiDistr_x_min_,
                                       hiDistr_x_max_,
                                       hiDistr_y_nbin_,
                                       hiDistr_y_min_,
                                       hiDistr_y_max_);

  hiDistrNoise2Min2D_->setAxisTitle("i#phi ", 2);
  hiDistrNoise2Min2D_->setAxisTitle("i#eta ", 1);

  // Variance
  hiDistrVarMBMin2D_ = ibooker.book2D("VarMBdepthMin1",
                                      "iphi- +ieta signal distribution at depth1",
                                      hiDistr_x_nbin_,
                                      hiDistr_x_min_,
                                      hiDistr_x_max_,
                                      hiDistr_y_nbin_,
                                      hiDistr_y_min_,
                                      hiDistr_y_max_);

  hiDistrVarMBMin2D_->setAxisTitle("i#phi ", 2);
  hiDistrVarMBMin2D_->setAxisTitle("i#eta ", 1);

  hiDistrVarNoiseMin2D_ = ibooker.book2D("VarNoisedepthMin1",
                                         "iphi-ieta noise distribution at depth1",
                                         hiDistr_x_nbin_,
                                         hiDistr_x_min_,
                                         hiDistr_x_max_,
                                         hiDistr_y_nbin_,
                                         hiDistr_y_min_,
                                         hiDistr_y_max_);

  hiDistrVarNoiseMin2D_->setAxisTitle("i#phi ", 2);
  hiDistrVarNoiseMin2D_->setAxisTitle("i#eta ", 1);
}

//--------------------------------------------------------
// void DQMHcalPhiSymAlCaReco::beginRun(const edm::Run& r, const EventSetup&
// context) {
////   eventCounter_ = 0;
//}

//--------------------------------------------------------

//-------------------------------------------------------------

void DQMHcalPhiSymAlCaReco::analyze(const Event &iEvent, const EventSetup &iSetup) {
  eventCounter_++;

  edm::Handle<FEDRawDataCollection> rawIn;
  iEvent.getByToken(rawInLabel_, rawIn);

  if (!rawIn.isValid()) {
    LogDebug("") << "HcalCalibAlgos: Error! can't get hbhe product!" << std::endl;
    return;
  }

  // get HCAL FEDs:
  std::vector<int> selFEDs;
  for (int i = FEDNumbering::MINHCALFEDID; i <= FEDNumbering::MAXHCALFEDID; i++) {
    selFEDs.push_back(i);
  }

  //  std::cout<<" Size of FED "<<selFEDs.size()<<std::endl;

  const FEDRawDataCollection *rdc = rawIn.product();

  bool hcalIsZS = false;
  int lvl1ID = 0;
  bool lvl1IDFound = false;
  for (unsigned int k = 0; k < selFEDs.size(); k++) {
    const FEDRawData &fedData = rdc->FEDData(selFEDs[k]);
    // std::cout<<fedData.size()*std::pow(1024.,-1)<<std::endl;
    hFEDsize->Fill(fedData.size() * std::pow(1024., -1), 1);

    // get HCAL DCC Header for each FEDRawData
    const HcalDCCHeader *dccHeader = (const HcalDCCHeader *)(fedData.data());
    if (dccHeader) {
      // walk through the HTR data...
      HcalHTRData htr;

      int nspigot = 0;
      for (int spigot = 0; spigot < HcalDCCHeader::SPIGOT_COUNT; spigot++) {
        nspigot++;

        if (!dccHeader->getSpigotPresent(spigot))
          continue;

        // Load the given decoder with the pointer and length from this spigot.
        dccHeader->getSpigotData(spigot, htr, fedData.size());

        if (k != 20 && nspigot != 14) {
          if (!htr.isUnsuppressed()) {
            hcalIsZS = true;
          }
        }
      }
    }

    // try to get the lvl1ID from the HCAL fed
    if (!lvl1IDFound && (fedData.size() > 0)) {
      // get FED Header for FEDRawData
      FEDHeader fedHeader(fedData.data());
      lvl1ID = fedHeader.lvl1ID();
      lvl1IDFound = true;
    }
  }  // loop over HcalFEDs

  hHcalIsZS->Fill(hcalIsZS);
  hL1Id->Fill(lvl1ID % period_);

  edm::Handle<HBHERecHitCollection> hbheNS;
  iEvent.getByToken(hbherecoNoise, hbheNS);

  if (!hbheNS.isValid()) {
    LogDebug("") << "HcalCalibAlgos: Error! can't get hbhe product!" << std::endl;
    return;
  }

  edm::Handle<HBHERecHitCollection> hbheMB;
  iEvent.getByToken(hbherecoMB, hbheMB);

  if (!hbheMB.isValid()) {
    LogDebug("") << "HcalCalibAlgos: Error! can't get hbhe product!" << std::endl;
    return;
  }

  edm::Handle<HFRecHitCollection> hfNS;
  iEvent.getByToken(hfrecoNoise, hfNS);

  if (!hfNS.isValid()) {
    LogDebug("") << "HcalCalibAlgos: Error! can't get hbhe product!" << std::endl;
    return;
  }

  edm::Handle<HFRecHitCollection> hfMB;
  iEvent.getByToken(hfrecoMB, hfMB);

  if (!hfMB.isValid()) {
    LogDebug("") << "HcalCalibAlgos: Error! can't get hbhe product!" << std::endl;
    return;
  }

  const HBHERecHitCollection HithbheNS = *(hbheNS.product());

  hiDistrHBHEsize1D_->Fill(HithbheNS.size() / ihbhe_size_);

  for (HBHERecHitCollection::const_iterator hbheItr = HithbheNS.begin(); hbheItr != HithbheNS.end(); hbheItr++) {
    DetId id = (*hbheItr).detid();
    HcalDetId hid = HcalDetId(id);

    if (hid.depth() == 1) {
      if (hid.ieta() > 0) {
        hiDistrNoisePl2D_->Fill(hid.ieta(), hid.iphi(), hbheItr->energy());
        hiDistrNoise2Pl2D_->Fill(hid.ieta(), hid.iphi(), hbheItr->energy() * hbheItr->energy());
      } else {
        hiDistrNoiseMin2D_->Fill(fabs(hid.ieta()), hid.iphi(), hbheItr->energy());
        hiDistrNoise2Min2D_->Fill(fabs(hid.ieta()), hid.iphi(), hbheItr->energy() * hbheItr->energy());
      }
    }
  }

  const HBHERecHitCollection HithbheMB = *(hbheMB.product());

  for (HBHERecHitCollection::const_iterator hbheItr = HithbheMB.begin(); hbheItr != HithbheMB.end(); hbheItr++) {
    DetId id = (*hbheItr).detid();
    HcalDetId hid = HcalDetId(id);

    if (hid.depth() == 1) {
      if (hid.ieta() > 0) {
        hiDistrMBPl2D_->Fill(hid.ieta(), hid.iphi(), hbheItr->energy());
        hiDistrMB2Pl2D_->Fill(hid.ieta(), hid.iphi(), hbheItr->energy() * hbheItr->energy());
      } else {
        hiDistrMBMin2D_->Fill(fabs(hid.ieta()), hid.iphi(), hbheItr->energy());
        hiDistrMB2Min2D_->Fill(fabs(hid.ieta()), hid.iphi(), hbheItr->energy() * hbheItr->energy());
      }
    }
  }

  const HFRecHitCollection HithfNS = *(hfNS.product());

  hiDistrHFsize1D_->Fill(HithfNS.size() / ihf_size_);

  for (HFRecHitCollection::const_iterator hbheItr = HithfNS.begin(); hbheItr != HithfNS.end(); hbheItr++) {
    DetId id = (*hbheItr).detid();
    HcalDetId hid = HcalDetId(id);

    if (hid.depth() == 1) {
      if (hid.ieta() > 0) {
        hiDistrNoisePl2D_->Fill(hid.ieta(), hid.iphi(), hbheItr->energy());
        hiDistrNoise2Pl2D_->Fill(hid.ieta(), hid.iphi(), hbheItr->energy() * hbheItr->energy());
      } else {
        hiDistrNoiseMin2D_->Fill(fabs(hid.ieta()), hid.iphi(), hbheItr->energy());
        hiDistrNoise2Min2D_->Fill(fabs(hid.ieta()), hid.iphi(), hbheItr->energy() * hbheItr->energy());
      }
    }
  }

  const HFRecHitCollection HithfMB = *(hfMB.product());

  for (HFRecHitCollection::const_iterator hbheItr = HithfMB.begin(); hbheItr != HithfMB.end(); hbheItr++) {
    DetId id = (*hbheItr).detid();
    HcalDetId hid = HcalDetId(id);

    if (hid.depth() == 1) {
      if (hid.ieta() > 0) {
        hiDistrMBPl2D_->Fill(hid.ieta(), hid.iphi(), hbheItr->energy());
        hiDistrMB2Pl2D_->Fill(hid.ieta(), hid.iphi(), hbheItr->energy() * hbheItr->energy());
      } else {
        hiDistrMBMin2D_->Fill(fabs(hid.ieta()), hid.iphi(), hbheItr->energy());
        hiDistrMB2Min2D_->Fill(fabs(hid.ieta()), hid.iphi(), hbheItr->energy() * hbheItr->energy());
      }
    }
  }

}  // analyze

//--------------------------------------------------------
//--------------------------------------------------------
void DQMHcalPhiSymAlCaReco::dqmEndRun(const Run &r, const EventSetup &context) {
  // Keep Variances
  if (eventCounter_ > 0 && !perLSsaving_) {
    for (int k = 0; k <= hiDistr_x_nbin_; k++) {
      for (int j = 0; j <= hiDistr_y_nbin_; j++) {
        // First moment
        float cc1 = hiDistrMBPl2D_->getBinContent(k, j);
        cc1 = cc1 * 1. / eventCounter_;
        float cc2 = hiDistrNoisePl2D_->getBinContent(k, j);
        cc2 = cc2 * 1. / eventCounter_;
        float cc3 = hiDistrMBMin2D_->getBinContent(k, j);
        cc3 = cc3 * 1. / eventCounter_;
        float cc4 = hiDistrNoiseMin2D_->getBinContent(k, j);
        cc4 = cc4 * 1. / eventCounter_;
        // Second moment
        float cc11 = hiDistrMB2Pl2D_->getBinContent(k, j);
        cc11 = cc11 * 1. / eventCounter_;
        hiDistrVarMBPl2D_->setBinContent(k, j, cc11 - cc1 * cc1);
        float cc22 = hiDistrNoise2Pl2D_->getBinContent(k, j);
        cc22 = cc22 * 1. / eventCounter_;
        hiDistrVarNoisePl2D_->setBinContent(k, j, cc22 - cc2 * cc2);
        float cc33 = hiDistrMB2Min2D_->getBinContent(k, j);
        cc33 = cc33 * 1. / eventCounter_;
        hiDistrVarMBMin2D_->setBinContent(k, j, cc33 - cc3 * cc3);
        float cc44 = hiDistrNoise2Min2D_->getBinContent(k, j);
        cc44 = cc44 * 1. / eventCounter_;
        hiDistrVarNoiseMin2D_->setBinContent(k, j, cc44 - cc4 * cc4);
      }
    }
  }
}
