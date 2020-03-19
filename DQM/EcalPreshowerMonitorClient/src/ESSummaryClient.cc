#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "DQM/EcalPreshowerMonitorClient/interface/ESSummaryClient.h"

using namespace std;

ESSummaryClient::ESSummaryClient(const edm::ParameterSet &ps)
    : ESClient(ps), meReportSummary_(nullptr), meReportSummaryMap_(nullptr) {
  for (unsigned iZ(0); iZ != 2; ++iZ)
    for (unsigned iD(0); iD != 2; ++iD)
      meReportSummaryContents_[iZ][iD] = nullptr;
}

ESSummaryClient::~ESSummaryClient() {}

void ESSummaryClient::book(DQMStore::IBooker &_ibooker) {
  if (debug_)
    cout << "ESSummaryClient: setup" << endl;

  _ibooker.setCurrentFolder(prefixME_ + "/EventInfo");

  meReportSummary_ = _ibooker.bookFloat("reportSummary");

  _ibooker.setCurrentFolder(prefixME_ + "/EventInfo/reportSummaryContents");

  char histo[200];

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      int iz = (i == 0) ? 1 : -1;
      sprintf(histo, "EcalPreshower Z %d P %d", iz, j + 1);
      meReportSummaryContents_[i][j] = _ibooker.bookFloat(histo);
    }
  }

  _ibooker.setCurrentFolder(prefixME_ + "/EventInfo");

  meReportSummaryMap_ = _ibooker.book2D("reportSummaryMap", "reportSummaryMap", 80, 0.5, 80.5, 80, 0.5, 80.5);
  meReportSummaryMap_->setAxisTitle("Si X", 1);
  meReportSummaryMap_->setAxisTitle("Si Y", 2);
}

void ESSummaryClient::fillReportSummary(DQMStore::IGetter &_igetter) {
  for (int i = 0; i < 80; i++) {
    for (int j = 0; j < 80; j++) {
      meReportSummaryMap_->setBinContent(i + 1, j + 1, -1.);
    }
  }

  char histo[200];

  float nDI_FedErr[80][80];
  float DCC[80][80];
  float eCount;

  MonitorElement *me;

  for (int i = 0; i < 80; ++i)
    for (int j = 0; j < 80; ++j) {
      nDI_FedErr[i][j] = -1;
      DCC[i][j] = 0;
    }

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      int iz = (i == 0) ? 1 : -1;

      sprintf(histo, "ES Integrity Errors Z %d P %d", iz, j + 1);
      me = _igetter.get(prefixME_ + "/ESIntegrityTask/" + histo);
      if (me)
        for (int x = 0; x < 40; ++x)
          for (int y = 0; y < 40; ++y)
            nDI_FedErr[i * 40 + x][(1 - j) * 40 + y] = me->getBinContent(x + 1, y + 1);

      sprintf(histo, "ES Integrity Summary 1 Z %d P %d", iz, j + 1);
      me = _igetter.get(prefixME_ + "/ESIntegrityClient/" + histo);
      if (me)
        for (int x = 0; x < 40; ++x)
          for (int y = 0; y < 40; ++y)
            DCC[i * 40 + x][(1 - j) * 40 + y] = me->getBinContent(x + 1, y + 1);

      sprintf(histo, "ES RecHit 2D Occupancy Z %d P %d", iz, j + 1);
      me = _igetter.get(prefixME_ + "/ESOccupancyTask/" + histo);
      if (me)
        eCount = me->getBinContent(40, 40);
      else
        eCount = 1.;
    }
  }

  // The global-summary
  // ReportSummary Map
  //  ES+F  ES-F
  //  ES+R  ES-R
  float nValidChannels = 0;
  float nGlobalErrors = 0;
  float nValidChannelsES[2][2] = {};
  float nGlobalErrorsES[2][2] = {};

  for (int x = 0; x < 80; ++x) {
    if (eCount < 1)
      break;  // Fill reportSummaryMap after have 1 event
    for (int y = 0; y < 80; ++y) {
      int z = (x < 40) ? 0 : 1;
      int p = (y >= 40) ? 0 : 1;

      if (DCC[x][y] == 0.) {
        meReportSummaryMap_->setBinContent(x + 1, y + 1, -1.);
      } else {
        if (nDI_FedErr[x][y] >= 0) {
          meReportSummaryMap_->setBinContent(x + 1, y + 1, 1 - (nDI_FedErr[x][y] / eCount));

          nValidChannels++;
          nGlobalErrors += nDI_FedErr[x][y] / eCount;

          nValidChannelsES[z][p]++;
          nGlobalErrorsES[z][p] += nDI_FedErr[x][y] / eCount;
        } else {
          meReportSummaryMap_->setBinContent(x + 1, y + 1, -1.);
        }
      }
    }
  }

  for (unsigned iZ(0); iZ != 2; ++iZ) {
    for (unsigned iD(0); iD != 2; ++iD) {
      float reportSummaryES(-1.);
      if (nValidChannelsES[iZ][iD] != 0)
        reportSummaryES = 1. - nGlobalErrorsES[iZ][iD] / nValidChannelsES[iZ][iD];

      meReportSummaryContents_[iZ][iD]->Fill(reportSummaryES);
    }
  }

  float reportSummary(-1.);
  if (nValidChannels != 0)
    reportSummary = 1. - nGlobalErrors / nValidChannels;

  meReportSummary_->Fill(reportSummary);
}

void ESSummaryClient::endLumiAnalyze(DQMStore::IGetter &_igetter) {
  fillReportSummary(_igetter);

  // The following overwrites the report summary if LS-based Good Channel
  // Fraction histogram is available The source is turned off by default in
  // ESIntegrityTask

  MonitorElement *source(_igetter.get(prefixME_ + "/ESIntegrityTask/ES Good Channel Fraction"));
  if (!source)
    return;

  meReportSummary_->Fill(-1.0);
  for (unsigned iZ(0); iZ != 2; ++iZ)
    for (unsigned iD(0); iD != 2; ++iD)
      meReportSummaryContents_[iZ][iD]->Fill(-1.);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      meReportSummaryContents_[i][j]->Fill(source->getBinContent(i + 1, j + 1));

  meReportSummary_->Fill(source->getBinContent(3, 3));
}

void ESSummaryClient::endJobAnalyze(DQMStore::IGetter &_igetter) { fillReportSummary(_igetter); }
