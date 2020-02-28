// -*- C++ -*-
//
// Package:    ZfitterAnalyzer
// Class:      ZfitterAnalyzer
//
/**\class ZfitterAnalyzer ZfitterAnalyzer.cc
 Zfitter/ZfitterAnalyzer/src/ZfitterAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Vieri Candelise
//         Created:  Mon Jun 13 09:49:08 CEST 2011
//
//

// system include files
#include <memory>

// user include files
#include "DQM/Physics/src/EwkDQM.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "TH1.h"
#include "TMath.h"
#include <cmath>
#include <iostream>
#include <string>

Double_t mybw(Double_t *, Double_t *);
Double_t mygauss(Double_t *, Double_t *);

class EcalZmassClient : public DQMEDHarvester {
public:
  explicit EcalZmassClient(const edm::ParameterSet &);
  ~EcalZmassClient() override;

private:
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  // ----------member data ---------------------------

  std::string prefixME_;
};

EcalZmassClient::EcalZmassClient(const edm::ParameterSet &iConfig)
    : prefixME_(iConfig.getUntrackedParameter<std::string>("prefixME", "")) {}

EcalZmassClient::~EcalZmassClient() {}

void EcalZmassClient::dqmEndJob(DQMStore::IBooker &_ibooker, DQMStore::IGetter &_igetter) {
  MonitorElement *h_fitres1;
  MonitorElement *h_fitres1bis;
  MonitorElement *h_fitres1Chi2;
  MonitorElement *h_fitres2;
  MonitorElement *h_fitres2bis;
  MonitorElement *h_fitres2Chi2;
  MonitorElement *h_fitres3;
  MonitorElement *h_fitres3bis;
  MonitorElement *h_fitres3Chi2;

  _ibooker.setCurrentFolder(prefixME_ + "/Zmass");
  h_fitres1 = _ibooker.book1D("Gaussian mean WP80 EB-EB", "Gaussian mean WP80 EB-EB", 1, 0, 1);
  h_fitres1bis = _ibooker.book1D("Gaussian sigma WP80 EB-EB", "Gaussian sigma WP80 EB-EB", 1, 0, 1);
  h_fitres1Chi2 = _ibooker.book1D(
      "Gaussian Chi2 result over NDF  WP80 EB-EB", "Gaussian Chi2 result over NDF  WP80 EB-EB", 1, 0, 1);

  h_fitres3 = _ibooker.book1D("Gaussian mean WP80 EB-EE", "Gaussian mean result WP80 EB-EE", 1, 0, 1);
  h_fitres3bis = _ibooker.book1D("Gaussian sigma WP80 EB-EE", "Gaussian sigma WP80 EB-EE", 1, 0, 1);
  h_fitres3Chi2 =
      _ibooker.book1D("Gaussian Chi2 result over NDF WP80 EB-EE", "Gaussian Chi2 result over NDF WP80 EB-EE", 1, 0, 1);

  h_fitres2 = _ibooker.book1D("Gaussian mean WP80 EE-EE", "Gaussian mean WP80 EE-EE", 1, 0, 1);
  h_fitres2bis = _ibooker.book1D("Gaussian sigma WP80 EE-EE", "Gaussian sigma WP80 EE-EE", 1, 0, 1);
  h_fitres2Chi2 =
      _ibooker.book1D("Gaussian Chi2 result over NDF WP80 EE-EE", "Gaussian Chi2 result over NDF WP80 EE-EE", 1, 0, 1);

  LogTrace("EwkAnalyzer") << "Parameters initialization";

  MonitorElement *me1 = _igetter.get(prefixME_ + "/Zmass/Z peak - WP80 EB-EB");
  MonitorElement *me2 = _igetter.get(prefixME_ + "/Zmass/Z peak - WP80 EE-EE");
  MonitorElement *me3 = _igetter.get(prefixME_ + "/Zmass/Z peak - WP80 EB-EE");

  if (me1 != nullptr) {
    TH1F *B = me1->getTH1F();
    TH1F *R1 = h_fitres1->getTH1F();
    int division = B->GetNbinsX();
    float massMIN = B->GetBinLowEdge(1);
    float massMAX = B->GetBinLowEdge(division + 1);
    // float BIN_SIZE = B->GetBinWidth(1);

    TF1 *func = new TF1("mygauss", mygauss, massMIN, massMAX, 3);
    func->SetParameter(0, 1.0);
    func->SetParName(0, "const");
    func->SetParameter(1, 95.0);
    func->SetParName(1, "mean");
    func->SetParameter(2, 5.0);
    func->SetParName(2, "sigma");

    double stats[4];
    R1->GetStats(stats);
    float N = 0;
    float mean = 0;
    float sigma = 0;
    N = B->GetEntries();

    try {
      if (N != 0) {
        B->Fit("mygauss", "QR");
        mean = std::abs(func->GetParameter(1));
        sigma = std::abs(func->GetParError(1));
      }

      if (N == 0 || mean < 50 || mean > 100 || sigma <= 0 || sigma > 20) {
        N = 1;
        mean = 0;
        sigma = 0;
      }

    } catch (cms::Exception &e) {
      edm::LogError("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
      N = 1;
      mean = 40;
      sigma = 0;
    }

    stats[0] = N;
    stats[1] = N;
    stats[2] = mean * N;
    stats[3] = sigma * sigma * N + mean * mean * N;

    R1->SetEntries(N);
    R1->PutStats(stats);
  }
  /*******************************************************/
  if (me1 != nullptr) {
    TH1F *Bbis = me1->getTH1F();
    TH1F *R1bis = h_fitres1bis->getTH1F();
    int division = Bbis->GetNbinsX();
    float massMIN = Bbis->GetBinLowEdge(1);
    float massMAX = Bbis->GetBinLowEdge(division + 1);
    // float BIN_SIZE = B->GetBinWidth(1);

    TF1 *func = new TF1("mygauss", mygauss, massMIN, massMAX, 3);
    func->SetParameter(0, 1.0);
    func->SetParName(0, "const");
    func->SetParameter(1, 95.0);
    func->SetParName(1, "mean");
    func->SetParameter(2, 5.0);
    func->SetParName(2, "sigma");

    double stats[4];
    R1bis->GetStats(stats);
    float N = 0;
    float rms = 0;
    float rmsErr = 0;
    N = Bbis->GetEntries();

    try {
      if (N != 0) {
        Bbis->Fit("mygauss", "QR");
        rms = std::abs(func->GetParameter(2));
        rmsErr = std::abs(func->GetParError(2));
      }

      if (N == 0 || rms < 0 || rms > 50 || rmsErr <= 0 || rmsErr > 50) {
        N = 1;
        rms = 0;
        rmsErr = 0;
      }

    } catch (cms::Exception &e) {
      edm::LogError("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
      N = 1;
      rms = 40;
      rmsErr = 0;
    }

    stats[0] = N;
    stats[1] = N;
    stats[2] = rms * N;
    stats[3] = rmsErr * rmsErr * N + rms * rms * N;

    R1bis->SetEntries(N);
    R1bis->PutStats(stats);
  }
  /****************************************/

  if (me2 != nullptr) {
    TH1F *E = me2->getTH1F();
    TH1F *R2 = h_fitres2->getTH1F();
    int division = E->GetNbinsX();
    float massMIN = E->GetBinLowEdge(1);
    float massMAX = E->GetBinLowEdge(division + 1);
    // float BIN_SIZE = E->GetBinWidth(1);

    TF1 *func = new TF1("mygauss", mygauss, massMIN, massMAX, 3);
    func->SetParameter(0, 1.0);
    func->SetParName(0, "const");
    func->SetParameter(1, 95.0);
    func->SetParName(1, "mean");
    func->SetParameter(2, 5.0);
    func->SetParName(2, "sigma");

    double stats[4];
    R2->GetStats(stats);
    float N = 0;
    float mean = 0;
    float sigma = 0;
    N = E->GetEntries();

    try {
      if (N != 0) {
        E->Fit("mygauss", "QR");
        mean = std::abs(func->GetParameter(1));
        sigma = std::abs(func->GetParError(1));
      }

      if (N == 0 || mean < 50 || mean > 100 || sigma <= 0 || sigma > 20) {
        N = 1;
        mean = 0;
        sigma = 0;
      }

    } catch (cms::Exception &e) {
      edm::LogError("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
      N = 1;
      mean = 40;
      sigma = 0;
    }

    stats[0] = N;
    stats[1] = N;
    stats[2] = mean * N;
    stats[3] = sigma * sigma * N + mean * mean * N;

    R2->SetEntries(N);
    R2->PutStats(stats);
  }
  /**************************************************************************/

  if (me2 != nullptr) {
    TH1F *Ebis = me2->getTH1F();
    TH1F *R2bis = h_fitres2bis->getTH1F();
    int division = Ebis->GetNbinsX();
    float massMIN = Ebis->GetBinLowEdge(1);
    float massMAX = Ebis->GetBinLowEdge(division + 1);
    // float BIN_SIZE = B->GetBinWidth(1);

    TF1 *func = new TF1("mygauss", mygauss, massMIN, massMAX, 3);
    func->SetParameter(0, 1.0);
    func->SetParName(0, "const");
    func->SetParameter(1, 95.0);
    func->SetParName(1, "mean");
    func->SetParameter(2, 5.0);
    func->SetParName(2, "sigma");

    double stats[4];
    R2bis->GetStats(stats);
    float N = 0;
    float rms = 0;
    float rmsErr = 0;
    N = Ebis->GetEntries();

    try {
      if (N != 0) {
        Ebis->Fit("mygauss", "QR");
        rms = std::abs(func->GetParameter(2));
        rmsErr = std::abs(func->GetParError(2));
      }

      if (N == 0 || rms < 0 || rms > 50 || rmsErr <= 0 || rmsErr > 50) {
        N = 1;
        rms = 0;
        rmsErr = 0;
      }

    } catch (cms::Exception &e) {
      edm::LogError("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
      N = 1;
      rms = 40;
      rmsErr = 0;
    }

    stats[0] = N;
    stats[1] = N;
    stats[2] = rms * N;
    stats[3] = rmsErr * rmsErr * N + rms * rms * N;

    R2bis->SetEntries(N);
    R2bis->PutStats(stats);
  }
  /*********************************************************************************************/

  if (me3 != nullptr) {
    TH1F *R3 = h_fitres3->getTH1F();
    TH1F *M = me3->getTH1F();
    int division = M->GetNbinsX();
    float massMIN = M->GetBinLowEdge(1);
    float massMAX = M->GetBinLowEdge(division + 1);
    // float BIN_SIZE = M->GetBinWidth(1);

    TF1 *func = new TF1("mygauss", mygauss, massMIN, massMAX, 3);
    func->SetParameter(0, 1.0);
    func->SetParName(0, "const");
    func->SetParameter(1, 95.0);
    func->SetParName(1, "mean");
    func->SetParameter(2, 5.0);
    func->SetParName(2, "sigma");

    double stats[4];
    R3->GetStats(stats);
    float N = 0;
    float mean = 0;
    float sigma = 0;
    N = M->GetEntries();

    try {
      if (N != 0) {
        M->Fit("mygauss", "QR");
        mean = std::abs(func->GetParameter(1));
        sigma = std::abs(func->GetParError(1));
      }
      if (N == 0 || mean < 50 || mean > 100 || sigma <= 0 || sigma > 20) {
        N = 1;
        mean = 0;
        sigma = 0;
      }

    } catch (cms::Exception &e) {
      edm::LogError("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
      N = 1;
      mean = 40;
      sigma = 0;
    }

    stats[0] = N;
    stats[1] = N;
    stats[2] = mean * N;
    stats[3] = sigma * sigma * N + mean * mean * N;

    R3->SetEntries(N);
    R3->PutStats(stats);
  }
  /********************************************************************************/

  if (me3 != nullptr) {
    TH1F *Mbis = me3->getTH1F();
    TH1F *R3bis = h_fitres3bis->getTH1F();
    int division = Mbis->GetNbinsX();
    float massMIN = Mbis->GetBinLowEdge(1);
    float massMAX = Mbis->GetBinLowEdge(division + 1);
    // float BIN_SIZE = B->GetBinWidth(1);

    TF1 *func = new TF1("mygauss", mygauss, massMIN, massMAX, 3);
    func->SetParameter(0, 1.0);
    func->SetParName(0, "const");
    func->SetParameter(1, 95.0);
    func->SetParName(1, "mean");
    func->SetParameter(2, 5.0);
    func->SetParName(2, "sigma");

    double stats[4];
    R3bis->GetStats(stats);
    float N = 0;
    float rms = 0;
    float rmsErr = 0;
    N = Mbis->GetEntries();

    try {
      if (N != 0) {
        Mbis->Fit("mygauss", "QR");
        rms = std::abs(func->GetParameter(2));
        rmsErr = std::abs(func->GetParError(2));
      }

      if (N == 0 || rms < 0 || rms > 50 || rmsErr <= 0 || rmsErr > 50) {
        N = 1;
        rms = 0;
        rmsErr = 0;
      }

    } catch (cms::Exception &e) {
      edm::LogError("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
      N = 1;
      rms = 40;
      rmsErr = 0;
    }

    stats[0] = N;
    stats[1] = N;
    stats[2] = rms * N;
    stats[3] = rmsErr * rmsErr * N + rms * rms * N;

    R3bis->SetEntries(N);
    R3bis->PutStats(stats);
  }

  /*Chi2 */

  if (me1 != nullptr) {
    TH1F *C1 = me1->getTH1F();
    TH1F *S1 = h_fitres1Chi2->getTH1F();
    int division = C1->GetNbinsX();
    float massMIN = C1->GetBinLowEdge(1);
    float massMAX = C1->GetBinLowEdge(division + 1);
    // float BIN_SIZE = B->GetBinWidth(1);

    TF1 *func = new TF1("mygauss", mygauss, massMIN, massMAX, 3);
    func->SetParameter(0, 1.0);
    func->SetParName(0, "const");
    func->SetParameter(1, 95.0);
    func->SetParName(1, "mean");
    func->SetParameter(2, 5.0);
    func->SetParName(2, "sigma");

    double stats[4];
    S1->GetStats(stats);
    float N = 0;
    float Chi2 = 0;
    float NDF = 0;
    N = C1->GetEntries();

    try {
      if (N != 0) {
        C1->Fit("mygauss", "QR");
        if ((func->GetNDF() != 0)) {
          Chi2 = std::abs(func->GetChisquare()) / std::abs(func->GetNDF());
          NDF = 0.1;
        }
      }

      if (N == 0 || Chi2 < 0 || NDF < 0) {
        N = 1;
        Chi2 = 0;
        NDF = 0;
      }

    } catch (cms::Exception &e) {
      edm::LogError("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
      N = 1;
      Chi2 = 40;
      NDF = 0;
    }

    stats[0] = N;
    stats[1] = N;
    stats[2] = Chi2 * N;
    stats[3] = NDF * NDF * N + Chi2 * Chi2 * N;

    S1->SetEntries(N);
    S1->PutStats(stats);
  }
  /**********************************************/

  if (me2 != nullptr) {
    TH1F *C2 = me2->getTH1F();
    TH1F *S2 = h_fitres2Chi2->getTH1F();
    int division = C2->GetNbinsX();
    float massMIN = C2->GetBinLowEdge(1);
    float massMAX = C2->GetBinLowEdge(division + 1);
    // float BIN_SIZE = B->GetBinWidth(1);

    TF1 *func = new TF1("mygauss", mygauss, massMIN, massMAX, 3);
    func->SetParameter(0, 1.0);
    func->SetParName(0, "const");
    func->SetParameter(1, 95.0);
    func->SetParName(1, "mean");
    func->SetParameter(2, 5.0);
    func->SetParName(2, "sigma");

    double stats[4];
    S2->GetStats(stats);
    float N = 0;
    float Chi2 = 0;
    float NDF = 0;
    N = C2->GetEntries();

    try {
      if (N != 0) {
        C2->Fit("mygauss", "QR");
        if (func->GetNDF() != 0) {
          Chi2 = std::abs(func->GetChisquare()) / std::abs(func->GetNDF());
          NDF = 0.1;
        }
      }

      if (N == 0 || Chi2 < 0 || NDF < 0) {
        N = 1;
        Chi2 = 0;
        NDF = 0;
      }

    } catch (cms::Exception &e) {
      edm::LogError("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
      N = 1;
      Chi2 = 40;
      NDF = 0;
    }

    stats[0] = N;
    stats[1] = N;
    stats[2] = Chi2 * N;
    stats[3] = NDF * NDF * N + Chi2 * Chi2 * N;

    S2->SetEntries(N);
    S2->PutStats(stats);
  }
  /**************************************************************************/
  if (me3 != nullptr) {
    TH1F *C3 = me3->getTH1F();
    TH1F *S3 = h_fitres3Chi2->getTH1F();
    int division = C3->GetNbinsX();
    float massMIN = C3->GetBinLowEdge(1);
    float massMAX = C3->GetBinLowEdge(division + 1);
    // float BIN_SIZE = B->GetBinWidth(1);

    TF1 *func = new TF1("mygauss", mygauss, massMIN, massMAX, 3);
    func->SetParameter(0, 1.0);
    func->SetParName(0, "const");
    func->SetParameter(1, 95.0);
    func->SetParName(1, "mean");
    func->SetParameter(2, 5.0);
    func->SetParName(2, "sigma");

    double stats[4];
    S3->GetStats(stats);
    float N = 0;
    float Chi2 = 0;
    float NDF = 0;
    N = C3->GetEntries();

    try {
      if (N != 0) {
        C3->Fit("mygauss", "QR");
        if ((func->GetNDF() != 0)) {
          Chi2 = std::abs(func->GetChisquare()) / std::abs(func->GetNDF());
          NDF = 0.1;
        }
      }

      if (N == 0 || Chi2 < 0 || NDF < 0) {
        N = 1;
        Chi2 = 0;
        NDF = 0;
      }

    } catch (cms::Exception &e) {
      edm::LogError("ZFitter") << "[Zfitter]: Exception when fitting..." << e.what();
      N = 1;
      Chi2 = 40;
      NDF = 0;
    }

    stats[0] = N;
    stats[1] = N;
    stats[2] = Chi2 * N;
    stats[3] = NDF * NDF * N + Chi2 * Chi2 * N;

    S3->SetEntries(N);
    S3->PutStats(stats);
  }
}

// Breit-Wigner function
Double_t mybw(Double_t *x, Double_t *par) {
  Double_t arg1 = 14.0 / 22.0;                        // 2 over pi
  Double_t arg2 = par[1] * par[1] * par[2] * par[2];  // Gamma=par[2]  M=par[1]
  Double_t arg3 = ((x[0] * x[0]) - (par[1] * par[1])) * ((x[0] * x[0]) - (par[1] * par[1]));
  Double_t arg4 = x[0] * x[0] * x[0] * x[0] * ((par[2] * par[2]) / (par[1] * par[1]));
  return par[0] * arg1 * arg2 / (arg3 + arg4);
}

// Gaussian
Double_t mygauss(Double_t *x, Double_t *par) {
  Double_t arg = 0;
  if (par[2] < 0)
    par[2] = -par[2];  // par[2]: sigma
  if (par[2] != 0)
    arg = (x[0] - par[1]) / par[2];                                                        // par[1]: mean
  return par[0] * TMath::Exp(-0.5 * arg * arg) / (TMath::Sqrt(2 * TMath::Pi()) * par[2]);  // par[0] is constant
}

DEFINE_FWK_MODULE(EcalZmassClient);
