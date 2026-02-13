#include "ZGammaplusJetsPostProcessor.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

ZGammaplusJetsPostProcessor::ZGammaplusJetsPostProcessor(const edm::ParameterSet &pset) {
  subDir_ = pset.getUntrackedParameter<std::string>("subDir");
  isMuonTrgigger_ = pset.getUntrackedParameter<std::string>("IsMuonTrigger", "");
  isPhotonTrgigger_ = pset.getUntrackedParameter<std::string>("IsPhotonTrigger", "");
}

void ZGammaplusJetsPostProcessor::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  bool isMuonDir = false;
  bool isPhotonDir = false;

  TPRegexp patternMuon(isMuonTrgigger_);
  TPRegexp patternPhoton(isPhotonTrgigger_);

  // go to the directory to be processed
  if (igetter.dirExists(subDir_))
    ibooker.cd(subDir_);
  else {
    edm::LogWarning("ZGammaplusJetsPostProcessor: ") << "Directory " << subDir_ << " not found, skip";
    return;
  }

  std::vector<std::string> subdirectories = igetter.getSubdirs();
  for (std::vector<std::string>::iterator dir = subdirectories.begin(); dir != subdirectories.end(); dir++) {
    ibooker.cd(*dir);

    isMuonDir = false;
    isPhotonDir = false;

    if (TString(*dir).Contains(patternMuon))
      isMuonDir = true;
    if (TString(*dir).Contains(patternPhoton))
      isPhotonDir = true;

    if (isMuonDir) {
      // Direct balance vs Z Pt
      histos_(ibooker, igetter, "DirectBalanceVsZPt_numerator", "meanDBvsZPt", "Z Pt", "mean Direct Balance vs Z Pt");
    }

    if (isPhotonDir) {
      // Direct balance vs Photon Pt
      histos_(ibooker,
              igetter,
              "DirectBalanceVsPhotonPt_numerator",
              "meanDBvsPhotonPt",
              "Photon Pt",
              "mean Direct Balance vs Photon Pt");
    }

    ibooker.goUp();
  }
}

TH1F *ZGammaplusJetsPostProcessor::histos_(DQMStore::IBooker &ibooker,
                                           DQMStore::IGetter &igetter,
                                           const std::string &DBName,
                                           const std::string &outName,
                                           const std::string &label,
                                           const std::string &title) {
  TH2F *h_DirectBalance = getHistogram(ibooker, igetter, ibooker.pwd() + "/" + DBName);

  if (h_DirectBalance == nullptr)
    edm::LogWarning("ZGammaplusJetsPostProcessor")
        << "DirectBalance histogram " << ibooker.pwd() + "/" + DBName << " does not exist";

  // Check if histograms actually exist
  if (!h_DirectBalance)
    return nullptr;

  int DB_nbins = h_DirectBalance->GetXaxis()->GetNbins();

  float bins[DB_nbins + 1];
  for (int i = 0; i <= (h_DirectBalance->GetXaxis()->GetNbins()) + 1; i++) {
    double xlow = h_DirectBalance->GetXaxis()->GetBinLowEdge(i);
    bins[i] = xlow;
  }

  MonitorElement *meanDBvsRefPt = ibooker.book1D(outName, title, DB_nbins + 1, bins);

  TH1F *pr = meanDBvsRefPt->getTH1F();

  // Enable the approximation
  TProfile::Approximate(kTRUE);
  TProfile *hp = h_DirectBalance->ProfileX("_pfx", 1, -1, "");  //"" --> error of the mean of all y values

  pr->SetTitle(TString::Format("%s (Profile X)", h_DirectBalance->GetTitle()));
  pr->GetXaxis()->SetTitle(label.c_str());
  pr->GetYaxis()->SetRangeUser(-3.99, 5.99);
  pr->SetYTitle("<mean Direct Balance>");
  pr->SetOption("PE");
  pr->SetLineColor(2);
  pr->SetLineWidth(2);
  pr->SetMarkerStyle(20);
  pr->SetMarkerSize(0.8);
  pr->SetStats(kTRUE);

  for (int i = 1; i <= hp->GetNbinsX() + 1; i++) {
    pr->SetBinContent(i + 1, hp->GetBinContent(i));
    if (hp->GetBinError(i) != 0.)
      pr->SetBinError(i + 1, hp->GetBinError(i));
  }

  return pr;
}

TH2F *ZGammaplusJetsPostProcessor::getHistogram(DQMStore::IBooker &ibooker,
                                                DQMStore::IGetter &igetter,
                                                const std::string &histoPath) {
  ibooker.pwd();
  MonitorElement *monElement = igetter.get(histoPath);
  if (monElement != nullptr) {
    if (monElement->getTH2F()->GetEntries() == 0) {
      return nullptr;
    } else {
      return monElement->getTH2F();
    }
  } else {
    return nullptr;
  }
}

DEFINE_FWK_MODULE(ZGammaplusJetsPostProcessor);
