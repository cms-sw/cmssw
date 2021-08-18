#include "DQMOffline/L1Trigger/interface/L1TDiffHarvesting.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace dqmoffline {
  namespace l1t {

    L1TDiffHarvesting::L1TDiffPlotHandler::L1TDiffPlotHandler(const edm::ParameterSet &ps, std::string plotName)
        : dir1_(ps.getUntrackedParameter<std::string>("dir1")),
          dir2_(ps.getUntrackedParameter<std::string>("dir2")),
          outputDir_(ps.getUntrackedParameter<std::string>("outputDir", dir1_)),
          plotName_(plotName),
          h1_(),
          h2_(),
          h_diff_(),
          histType1_(),
          histType2_() {}

    L1TDiffHarvesting::L1TDiffPlotHandler::L1TDiffPlotHandler(const L1TDiffHarvesting::L1TDiffPlotHandler &handler)
        : dir1_(handler.dir1_),
          dir2_(handler.dir2_),
          outputDir_(handler.outputDir_),
          plotName_(handler.plotName_),
          h1_(),
          h2_(),
          h_diff_(),
          histType1_(),
          histType2_() {}

    void L1TDiffHarvesting::L1TDiffPlotHandler::computeDiff(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
      loadHistograms(igetter);
      if (!isValid()) {
        return;
      }
      bookDiff(ibooker);

      TH1 *h_diff;
      TH1 *h1;
      TH1 *h2;
      bool is1D(histType1_ == MonitorElement::Kind::TH1F || histType1_ == MonitorElement::Kind::TH1D);
      bool is2D(histType1_ == MonitorElement::Kind::TH2F || histType1_ == MonitorElement::Kind::TH2D);
      bool isProfile(histType1_ == MonitorElement::Kind::TPROFILE);

      if (is1D) {
        h_diff = h_diff_->getTH1F();
        h1 = h1_->getTH1F();
        h2 = h2_->getTH1F();
      } else if (is2D) {
        h_diff = h_diff_->getTH2F();
        h1 = h1_->getTH2F();
        h2 = h2_->getTH2F();
      } else if (isProfile) {
        h_diff = h_diff_->getTProfile();
        h1 = h1_->getTProfile();
        h2 = h2_->getTProfile();
      } else {
        edm::LogWarning("L1TDiffHarvesting::L1TDiffPlotHandler::computeDiff")
            << "Unknown histogram type. Quitting booking" << std::endl;

        return;
      }
      h_diff->Add(h1);
      h_diff->Add(h2, -1);
      // if histograms are identical h_diff will have 0 entries -> not good to check if anything happened
      // let's fix it
      h_diff->SetEntries(h1->GetEntries() + h2->GetEntries());
    }

    void L1TDiffHarvesting::L1TDiffPlotHandler::loadHistograms(DQMStore::IGetter &igetter) {
      std::string h1Name = dir1_ + "/" + plotName_;
      std::string h2Name = dir2_ + "/" + plotName_;
      h1_ = igetter.get(h1Name);
      h2_ = igetter.get(h2Name);

      if (!h1_ || !h2_) {
        edm::LogWarning("L1TDiffHarvesting::L1TDiffPlotHandler::loadHistograms")
            << (!h1_ && !h2_ ? h1Name + " && " + h2Name
                : !h1_       ? h1Name
                             : h2Name)
            << " not gettable. Quitting booking" << std::endl;

        return;
      }

      histType1_ = h1_->kind();
      histType2_ = h2_->kind();
    }

    bool L1TDiffHarvesting::L1TDiffPlotHandler::isValid() const {
      if (histType1_ == MonitorElement::Kind::INVALID) {
        edm::LogWarning("L1TDiffHarvesting::L1TDiffPlotHandler::isValid")
            << " Could not find a supported histogram type" << std::endl;
        return false;
      }
      if (histType1_ != histType2_) {
        edm::LogWarning("L1TDiffHarvesting::L1TDiffPlotHandler::isValid")
            << " Histogram 1 and 2 have different histogram types" << std::endl;
        return false;
      }
      return true;
    }

    void L1TDiffHarvesting::L1TDiffPlotHandler::bookDiff(DQMStore::IBooker &ibooker) {
      ibooker.setCurrentFolder(outputDir_);

      bool is1D(histType1_ == MonitorElement::Kind::TH1F || histType1_ == MonitorElement::Kind::TH1D);
      bool is2D(histType1_ == MonitorElement::Kind::TH2F || histType1_ == MonitorElement::Kind::TH2D);
      bool isProfile(histType1_ == MonitorElement::Kind::TPROFILE);

      if (is1D) {
        TH1F *h1 = h1_->getTH1F();
        double min = h1->GetXaxis()->GetXmin();
        double max = h1->GetXaxis()->GetXmax();
        int nBins = h1->GetNbinsX();
        h_diff_ = ibooker.book1D(plotName_, plotName_, nBins, min, max);
      } else if (is2D) {
        TH2F *h1 = h1_->getTH2F();
        double minX = h1->GetXaxis()->GetXmin();
        double maxX = h1->GetXaxis()->GetXmax();
        double minY = h1->GetYaxis()->GetXmin();
        double maxY = h1->GetYaxis()->GetXmax();
        int nBinsX = h1->GetNbinsX();
        int nBinsY = h1->GetNbinsY();

        h_diff_ = ibooker.book2D(plotName_, plotName_, nBinsX, minX, maxX, nBinsY, minY, maxY);
      } else if (isProfile) {
        TProfile *h1 = h1_->getTProfile();
        double minX = h1->GetXaxis()->GetXmin();
        double maxX = h1->GetXaxis()->GetXmax();
        double minY = h1->GetYaxis()->GetXmin();
        double maxY = h1->GetYaxis()->GetXmax();
        int nBins = h1->GetNbinsX();
        h_diff_ = ibooker.bookProfile(plotName_, plotName_, nBins, minX, maxX, minY, maxY);
      } else {
        edm::LogWarning("L1TDiffHarvesting::L1TDiffPlotHandler::bookDiff")
            << "Unknown histogram type. Quitting booking" << std::endl;

        return;
      }
    }

    L1TDiffHarvesting::L1TDiffHarvesting(const edm::ParameterSet &ps) : plotHandlers_() {
      using namespace std;
      for (const auto &plotConfig : ps.getUntrackedParameter<std::vector<edm::ParameterSet>>("plotCfgs")) {
        vector<string> plots = plotConfig.getUntrackedParameter<vector<string>>("plots");
        for (const auto &plot : plots) {
          plotHandlers_.push_back(L1TDiffPlotHandler(plotConfig, plot));
        }
      }
    }

    L1TDiffHarvesting::~L1TDiffHarvesting() {}

    void L1TDiffHarvesting::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
      edm::LogInfo("L1TEfficiencyHarvesting") << "Called endRun." << std::endl;

      for (auto plotHandler : plotHandlers_) {
        plotHandler.computeDiff(ibooker, igetter);
      }
    }

    DEFINE_FWK_MODULE(L1TDiffHarvesting);

  }  // namespace l1t
}  // namespace dqmoffline
