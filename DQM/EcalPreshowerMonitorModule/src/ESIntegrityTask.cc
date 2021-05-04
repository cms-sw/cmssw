#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRawData/interface/ESDCCHeaderBlock.h"
#include "DataFormats/EcalRawData/interface/ESKCHIPBlock.h"

#include "DQM/EcalPreshowerMonitorModule/interface/ESIntegrityTask.h"

using namespace cms;
using namespace edm;
using namespace std;

ESIntegrityTask::ESIntegrityTask(const ParameterSet& ps) {
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");
  lookup_ = ps.getUntrackedParameter<FileInPath>("LookupTable");

  dccCollections_ = consumes<ESRawDataCollection>(ps.getParameter<InputTag>("ESDCCCollections"));
  kchipCollections_ = consumes<ESLocalRawDataCollection>(ps.getParameter<InputTag>("ESKChipCollections"));

  doLumiAnalysis_ = ps.getParameter<bool>("DoLumiAnalysis");

  // read in look-up table
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 40; ++k)
        for (int m = 0; m < 40; ++m) {
          fed_[i][j][k][m] = -1;
          kchip_[i][j][k][m] = -1;
          fiber_[i][j][k][m] = -1;
        }

  int nLines_, z, iz, ip, ix, iy, fed, kchip, pace, bundle, fiber, optorx;
  ifstream file;
  file.open(lookup_.fullPath().c_str());
  if (file.is_open()) {
    file >> nLines_;

    for (int i = 0; i < nLines_; ++i) {
      file >> iz >> ip >> ix >> iy >> fed >> kchip >> pace >> bundle >> fiber >> optorx;

      z = (iz == -1) ? 2 : iz;
      fed_[z - 1][ip - 1][ix - 1][iy - 1] = fed;
      kchip_[z - 1][ip - 1][ix - 1][iy - 1] = kchip;
      fiber_[z - 1][ip - 1][ix - 1][iy - 1] = (fiber - 1) + (optorx - 1) * 12;
    }
  } else {
    cout << "ESIntegrityTask : Look up table file can not be found in " << lookup_.fullPath().c_str() << endl;
  }

  ievt_ = 0;
}

void ESIntegrityTask::dqmEndRun(const Run& r, const EventSetup& c) {
  // In case of Lumi based analysis Disable SoftReset from Integrity histogram to get full statistics
  // TODO: no longer possible, clone histo beforehand if full statisticcs at end of run are required.
}

std::shared_ptr<ESLSCache> ESIntegrityTask::globalBeginLuminosityBlock(const edm::LuminosityBlock& lumi,
                                                                       const edm::EventSetup& c) const {
  LogInfo("ESIntegrityTask") << "analyzed " << ievt_ << " events";
  // In case of Lumi based analysis SoftReset the Integrity histogram
  auto lumiCache = std::make_shared<ESLSCache>();
  lumiCache->ievtLS_ = 0;
  if (doLumiAnalysis_) {
    for (int iz = 0; iz < 2; ++iz) {
      for (int ip = 0; ip < 2; ++ip) {
        for (int ix = 0; ix < 40; ++ix) {
          for (int iy = 0; iy < 40; ++iy) {
            (lumiCache->DIErrorsLS_)[iz][ip][ix][iy] = 0;
          }
        }
      }
    }
  }
  return lumiCache;
}

void ESIntegrityTask::globalEndLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& c) {
  if (doLumiAnalysis_)
    calculateDIFraction(lumi, c);
}

void ESIntegrityTask::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&) {
  char histo[200];

  iBooker.setCurrentFolder(prefixME_ + "/ESIntegrityTask");

  sprintf(histo, "ES FEDs used for data taking");
  meFED_ = iBooker.book1D(histo, histo, 56, 519.5, 575.5);
  meFED_->setAxisTitle("ES FED", 1);
  meFED_->setAxisTitle("Num of Events", 2);

  sprintf(histo, "ES Gain used for data taking");
  meGain_ = iBooker.book1D(histo, histo, 2, -0.5, 1.5);
  meGain_->setAxisTitle("ES Gain", 1);
  meGain_->setAxisTitle("Num of Events", 2);

  sprintf(histo, "ES DCC Error codes");
  meDCCErr_ = iBooker.book2D(histo, histo, 56, 519.5, 575.5, 6, -0.5, 5.5);
  meDCCErr_->setAxisTitle("ES FED", 1);
  meDCCErr_->setAxisTitle("ES DCC Error code", 2);

  sprintf(histo, "ES SLink CRC Errors");
  meSLinkCRCErr_ = iBooker.book1D(histo, histo, 56, 519.5, 575.5);
  meSLinkCRCErr_->setAxisTitle("ES FED", 1);
  meSLinkCRCErr_->setAxisTitle("Num of Events", 2);

  sprintf(histo, "ES DCC CRC Errors");
  meDCCCRCErr_ = iBooker.book2D(histo, histo, 56, 519.5, 575.5, 3, -0.5, 2.5);
  meDCCCRCErr_->setAxisTitle("ES FED", 1);
  meDCCCRCErr_->setAxisTitle("ES OptoRX", 2);

  sprintf(histo, "ES OptoRX used for data taking");
  meOptoRX_ = iBooker.book2D(histo, histo, 56, 519.5, 575.5, 3, -0.5, 2.5);
  meOptoRX_->setAxisTitle("ES FED", 1);
  meOptoRX_->setAxisTitle("ES OptoRX", 2);

  sprintf(histo, "ES OptoRX BC mismatch");
  meOptoBC_ = iBooker.book2D(histo, histo, 56, 519.5, 575.5, 3, -0.5, 2.5);
  meOptoBC_->setAxisTitle("ES FED", 1);
  meOptoBC_->setAxisTitle("ES OptoRX", 2);

  sprintf(histo, "ES Fiber Bad Status");
  meFiberBadStatus_ = iBooker.book2D(histo, histo, 56, 519.5, 575.5, 36, 0.5, 36.5);
  meFiberBadStatus_->setAxisTitle("ES FED", 1);
  meFiberBadStatus_->setAxisTitle("Fiber Number", 2);

  sprintf(histo, "ES Fiber Error Code");
  meFiberErrCode_ = iBooker.book1D(histo, histo, 17, -0.5, 16.5);
  meFiberErrCode_->setAxisTitle("Fiber Error Code", 1);

  sprintf(histo, "ES Fiber Off");

  meFiberOff_ = iBooker.book2D(histo, histo, 56, 519.5, 575.5, 36, 0.5, 36.5);
  meFiberOff_->setAxisTitle("ES FED", 1);
  meFiberOff_->setAxisTitle("Fiber Number", 2);

  sprintf(histo, "ES Warning Event Dropped");
  meEVDR_ = iBooker.book2D(histo, histo, 56, 519.5, 575.5, 36, 0.5, 36.5);
  meEVDR_->setAxisTitle("ES FED", 1);
  meEVDR_->setAxisTitle("Fiber Number", 2);

  sprintf(histo, "ES KChip Flag 1 Error codes");
  meKF1_ = iBooker.book2D(histo, histo, 1550, -0.5, 1549.5, 16, -0.5, 15.5);
  meKF1_->setAxisTitle("ES KChip", 1);
  meKF1_->setAxisTitle("ES KChip F1 Error Code ", 2);

  sprintf(histo, "ES KChip Flag 2 Error codes");
  meKF2_ = iBooker.book2D(histo, histo, 1550, -0.5, 1549.5, 16, -0.5, 15.5);
  meKF2_->setAxisTitle("ES KChip", 1);
  meKF2_->setAxisTitle("ES KChip F1 Error Code ", 2);

  sprintf(histo, "ES KChip BC mismatch with OptoRX");
  meKBC_ = iBooker.book1D(histo, histo, 1550, -0.5, 1549.5);
  meKBC_->setAxisTitle("ES KChip", 1);
  meKBC_->setAxisTitle("Num of BC mismatch", 2);

  sprintf(histo, "ES KChip EC mismatch with OptoRX");
  meKEC_ = iBooker.book1D(histo, histo, 1550, -0.5, 1549.5);
  meKEC_->setAxisTitle("ES KChip", 1);
  meKEC_->setAxisTitle("Num of EC mismatch", 2);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      int iz = (i == 0) ? 1 : -1;
      sprintf(histo, "ES Integrity Errors Z %d P %d", iz, j + 1);
      meDIErrors_[i][j] = iBooker.book2D(histo, histo, 40, 0.5, 40.5, 40, 0.5, 40.5);
      meDIErrors_[i][j]->setAxisTitle("Si X", 1);
      meDIErrors_[i][j]->setAxisTitle("Si Y", 2);
    }

  if (doLumiAnalysis_) {
    sprintf(histo, "ES Good Channel Fraction");
    meDIFraction_ = iBooker.book2D(histo, histo, 3, 1.0, 3.0, 3, 1.0, 3.0);
  }
}

void ESIntegrityTask::endJob(void) { LogInfo("ESIntegrityTask") << "analyzed " << ievt_ << " events"; }

void ESIntegrityTask::analyze(const Event& e, const EventSetup& c) {
  ievt_++;
  auto lumiCache = luminosityBlockCache(e.getLuminosityBlock().index());
  ++(lumiCache->ievtLS_);

  Handle<ESRawDataCollection> dccs;
  Handle<ESLocalRawDataCollection> kchips;

  // Fill # of events
  meDCCErr_->Fill(575, 2, 1);
  meDCCCRCErr_->Fill(575, 2, 1);
  meOptoRX_->Fill(575, 2, 1);
  meOptoBC_->Fill(575, 2, 1);
  meFiberBadStatus_->Fill(575, 36, 1);
  meFiberOff_->Fill(575, 36, 1);
  meEVDR_->Fill(575, 36, 1);

  // # of DI errors
  Double_t nDIErr[56][36];
  for (int i = 0; i < 56; ++i)
    for (int j = 0; j < 36; ++j)
      nDIErr[i][j] = 0;

  // DCC
  vector<int> fiberStatus;
  if (e.getByToken(dccCollections_, dccs)) {
    for (ESRawDataCollection::const_iterator dccItr = dccs->begin(); dccItr != dccs->end(); ++dccItr) {
      ESDCCHeaderBlock dcc = (*dccItr);

      meFED_->Fill(dcc.fedId());

      meDCCErr_->Fill(dcc.fedId(), dcc.getDCCErrors());

      // SLink CRC error
      if (dcc.getDCCErrors() == 101) {
        meSLinkCRCErr_->Fill(dcc.fedId());
        for (int j = 0; j < 36; ++j)
          nDIErr[dcc.fedId() - 520][j]++;
      }

      if (dcc.getOptoRX0() == 129) {
        meOptoRX_->Fill(dcc.fedId(), 0);
        if (((dcc.getOptoBC0() - 15) & 0x0fff) != dcc.getBX())
          meOptoBC_->Fill(dcc.fedId(), 0);
      }
      if (dcc.getOptoRX1() == 129) {
        meOptoRX_->Fill(dcc.fedId(), 1);
        if (((dcc.getOptoBC1() - 15) & 0x0fff) != dcc.getBX())
          meOptoBC_->Fill(dcc.fedId(), 1);
      }
      if (dcc.getOptoRX2() == 129) {
        meOptoRX_->Fill(dcc.fedId(), 2);
        if (((dcc.getOptoBC2() - 15) & 0x0fff) != dcc.getBX())
          meOptoBC_->Fill(dcc.fedId(), 2);
      }

      if (dcc.getOptoRX0() == 128) {
        meDCCCRCErr_->Fill(dcc.fedId(), 0);
        for (int j = 0; j < 12; ++j)
          nDIErr[dcc.fedId() - 520][j]++;
      }
      if (dcc.getOptoRX1() == 128) {
        meDCCCRCErr_->Fill(dcc.fedId(), 1);
        for (int j = 12; j < 24; ++j)
          nDIErr[dcc.fedId() - 520][j]++;
      }
      if (dcc.getOptoRX2() == 128) {
        meDCCCRCErr_->Fill(dcc.fedId(), 2);
        for (int j = 24; j < 36; ++j)
          nDIErr[dcc.fedId() - 520][j]++;
      }

      fiberStatus = dcc.getFEChannelStatus();

      for (unsigned int i = 0; i < fiberStatus.size(); ++i) {
        if (fiberStatus[i] == 4 || fiberStatus[i] == 8 || fiberStatus[i] == 10 || fiberStatus[i] == 11 ||
            fiberStatus[i] == 12) {
          meFiberBadStatus_->Fill(dcc.fedId(), i + 1, 1);
          meFiberErrCode_->Fill(fiberStatus[i]);
          nDIErr[dcc.fedId() - 520][i]++;
        }
        if (fiberStatus[i] == 7)
          meFiberOff_->Fill(dcc.fedId(), i + 1, 1);
        if (fiberStatus[i] == 6) {
          meFiberErrCode_->Fill(fiberStatus[i]);
          meEVDR_->Fill(dcc.fedId(), i + 1, 1);
        }
      }

      runtype_ = dcc.getRunType();
      seqtype_ = dcc.getSeqType();
      dac_ = dcc.getDAC();
      gain_ = dcc.getGain();
      precision_ = dcc.getPrecision();

      meGain_->Fill(gain_);
    }
  } else {
    LogWarning("ESIntegrityTask") << "dccCollections not available";
  }

  // KCHIP's
  if (e.getByToken(kchipCollections_, kchips)) {
    for (ESLocalRawDataCollection::const_iterator kItr = kchips->begin(); kItr != kchips->end(); ++kItr) {
      ESKCHIPBlock kchip = (*kItr);

      meKF1_->Fill(kchip.id(), kchip.getFlag1());
      meKF2_->Fill(kchip.id(), kchip.getFlag2());
      if (kchip.getBC() != kchip.getOptoBC())
        meKBC_->Fill(kchip.id());
      if (kchip.getEC() != kchip.getOptoEC())
        meKEC_->Fill(kchip.id());
    }
  } else {
    LogWarning("ESIntegrityTask") << "kchipCollections not available";
  }

  // Fill # of DI errors
  for (int iz = 0; iz < 2; ++iz)
    for (int ip = 0; ip < 2; ++ip)
      for (int ix = 0; ix < 40; ++ix)
        for (int iy = 0; iy < 40; ++iy) {
          if (fed_[iz][ip][ix][iy] == -1)
            continue;

          if (nDIErr[fed_[iz][ip][ix][iy] - 520][fiber_[iz][ip][ix][iy]] > 0) {
            meDIErrors_[iz][ip]->Fill(ix + 1, iy + 1, 1);
            if (doLumiAnalysis_)
              (lumiCache->DIErrorsLS_)[iz][ip][ix][iy] += 1;
          }
        }
}
//
// -- Calculate Data Integrity Fraction
//
void ESIntegrityTask::calculateDIFraction(const edm::LuminosityBlock& lumi, const edm::EventSetup& c) {
  float nValidChannels = 0;
  float nGlobalErrors = 0;
  auto lumiCache = luminosityBlockCache(lumi.index());

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      float nValidChannelsES = 0;
      float nGlobalErrorsES = 0;
      float reportSummaryES = -1;
      for (int x = 0; x < 40; ++x) {
        for (int y = 0; y < 40; ++y) {
          float val = 1.0 * ((lumiCache->DIErrorsLS_)[i][j][x][y]);
          if (fed_[i][j][x][y] == -1)
            continue;
          if ((lumiCache->ievtLS_) != 0)
            nGlobalErrors += val / (lumiCache->ievtLS_);
          nValidChannels++;
          if ((lumiCache->ievtLS_) != 0)
            nGlobalErrorsES += val / (lumiCache->ievtLS_);
          nValidChannelsES++;
        }
      }
      if (nValidChannelsES != 0)
        reportSummaryES = 1 - nGlobalErrorsES / nValidChannelsES;
      meDIFraction_->setBinContent(i + 1, j + 1, reportSummaryES);
    }
  }
  float reportSummary = -1.0;
  if (nValidChannels != 0)
    reportSummary = 1.0 - nGlobalErrors / nValidChannels;
  meDIFraction_->setBinContent(3, 3, reportSummary);
}
DEFINE_FWK_MODULE(ESIntegrityTask);
