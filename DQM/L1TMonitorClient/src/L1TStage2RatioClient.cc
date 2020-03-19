#include "DQM/L1TMonitorClient/interface/L1TStage2RatioClient.h"

L1TStage2RatioClient::L1TStage2RatioClient(const edm::ParameterSet& ps)
    : monitorDir_(ps.getUntrackedParameter<std::string>("monitorDir")),
      inputNum_(ps.getUntrackedParameter<std::string>("inputNum")),
      inputDen_(ps.getUntrackedParameter<std::string>("inputDen")),
      ratioName_(ps.getUntrackedParameter<std::string>("ratioName")),
      ratioTitle_(ps.getUntrackedParameter<std::string>("ratioTitle")),
      yAxisTitle_(ps.getUntrackedParameter<std::string>("yAxisTitle")),
      binomialErr_(ps.getUntrackedParameter<bool>("binomialErr")),
      ignoreBin_(ps.getUntrackedParameter<std::vector<int>>("ignoreBin")),
      ratioME_(nullptr) {}

L1TStage2RatioClient::~L1TStage2RatioClient() {}

void L1TStage2RatioClient::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("monitorDir", "")
      ->setComment("Target directory in the DQM file. Will be created if not existing.");
  desc.addUntracked<std::string>("inputNum", "")->setComment("Path to numerator histogram.");
  desc.addUntracked<std::string>("inputDen", "")->setComment("Path to denominator histogram.");
  desc.addUntracked<std::string>("ratioName", "ratio")->setComment("Ratio plot name.");
  desc.addUntracked<std::string>("ratioTitle", "ratio")->setComment("Ratio plot title.");
  desc.addUntracked<std::string>("yAxisTitle", "")->setComment("Title of y axis.");
  desc.addUntracked<bool>("binomialErr", "true")->setComment("Compute binomial errors.");
  desc.addUntracked<std::vector<int>>("ignoreBin", std::vector<int>())
      ->setComment("List of bins to ignore. Will set their ratio to 0.");
  descriptions.add("l1TStage2RatioClient", desc);
}

void L1TStage2RatioClient::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                                                 DQMStore::IGetter& igetter,
                                                 const edm::LuminosityBlock& lumiSeg,
                                                 const edm::EventSetup& c) {
  book(ibooker, igetter);
  processHistograms(igetter);
}

void L1TStage2RatioClient::book(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  // Book when called the first time. Otherwise reset the ratio histogram.
  if (ratioME_ == nullptr) {
    ibooker.setCurrentFolder(monitorDir_);

    // get the axis range from the numerator histogram
    const MonitorElement* numME_ = igetter.get(inputNum_);
    if (numME_) {
      TH1F* hNum = numME_->getTH1F();

      ratioME_ = ibooker.book1D(
          ratioName_, ratioTitle_, hNum->GetNbinsX(), hNum->GetXaxis()->GetXmin(), hNum->GetXaxis()->GetXmax());
      ratioME_->setEfficiencyFlag();
      ratioME_->setAxisTitle(yAxisTitle_, 2);
    }
  } else {
    ratioME_->Reset();
  }
}

void L1TStage2RatioClient::processHistograms(DQMStore::IGetter& igetter) {
  const MonitorElement* numME_ = igetter.get(inputNum_);
  const MonitorElement* denME_ = igetter.get(inputDen_);

  if (numME_ && denME_) {
    TH1F* hNum = numME_->getTH1F();
    TH1F* hDen = dynamic_cast<TH1F*>(denME_->getTH1F()->Clone("den"));

    TH1F* hRatio = ratioME_->getTH1F();

    // Set the axis labels the same as the numerator histogram to be able to divide
    if (hNum->GetXaxis()->IsAlphanumeric()) {
      for (int i = 1; i <= hNum->GetNbinsX(); ++i) {
        hDen->GetXaxis()->SetBinLabel(i, hNum->GetXaxis()->GetBinLabel(i));
        hRatio->GetXaxis()->SetBinLabel(i, hNum->GetXaxis()->GetBinLabel(i));
      }
    }

    std::string errOption;
    if (binomialErr_) {
      errOption = "B";
    }

    hRatio->Divide(hNum, hDen, 1, 1, errOption.c_str());

    // Set the ratio to 0 for those bins that need to be ignored
    for (const int& bin : ignoreBin_) {
      if (bin > 0 && bin <= hRatio->GetNbinsX()) {
        hRatio->SetBinContent(bin, 0.0);
        hRatio->GetXaxis()->SetBinLabel(bin, "Ignored");
      }
    }

    delete hDen;
  }
}

void L1TStage2RatioClient::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  book(ibooker, igetter);
  processHistograms(igetter);
}
