/**
 * \file L1TEfficiencyHarvesting.cc
 *
 * \author J. Pela, C. Battilana
 *
 */

// L1TMonitor includes
#include "DQMOffline/L1Trigger/interface/L1TEfficiencyHarvesting.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace edm;
using namespace std;

namespace dqmoffline {
namespace l1t {
L1TEfficiencyPlotHandler::L1TEfficiencyPlotHandler(const ParameterSet & ps, std::string plotName) :
        numeratorDir_(ps.getUntrackedParameter < std::string > ("numeratorDir")),
        denominatorDir_(ps.getUntrackedParameter < std::string > ("denominatorDir", numeratorDir_)),
        outputDir_(ps.getUntrackedParameter < std::string > ("outputDir", numeratorDir_)),
        plotName_(plotName),
        numeratorSuffix_(ps.getUntrackedParameter < std::string > ("numeratorSuffix", "Num")),
        denominatorSuffix_(ps.getUntrackedParameter < std::string > ("denominatorSuffix", "Den")),
        h_efficiency_()
{

}

L1TEfficiencyPlotHandler::L1TEfficiencyPlotHandler(const L1TEfficiencyPlotHandler &handler) :
        numeratorDir_(handler.numeratorDir_),
        denominatorDir_(handler.denominatorDir_),
        outputDir_(handler.outputDir_),
        plotName_(handler.plotName_),
        numeratorSuffix_(handler.numeratorSuffix_),
        denominatorSuffix_(handler.denominatorSuffix_),
        h_efficiency_(handler.h_efficiency_)
{

}

void L1TEfficiencyPlotHandler::book(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter)
{

  edm::LogInfo("L1TEfficiencyPlotHandler") << "Booking efficiency histogram for " << outputDir_ << " and " << plotName_
      << endl;

  std::string numeratorName = numeratorDir_ + "/" + plotName_ + numeratorSuffix_;
  std::string denominatorName = denominatorDir_ + "/" + plotName_ + denominatorSuffix_;
  MonitorElement *num = igetter.get(numeratorName);
  MonitorElement *den = igetter.get(denominatorName);

  if (!num || !den) {

    edm::LogError("L1TEfficiencyPlotHandler")
        << (!num && !den ? numeratorName + " && " + denominatorName : !num ? numeratorName : denominatorName)
        << " not gettable. Quitting booking" << endl;
    return;
  }

  TH1F *numH = num->getTH1F();
  TH1F *denH = den->getTH1F();

  if (!numH || !denH) {

    edm::LogError("L1TEfficiencyPlotHandler")
        << (!numH && !denH ? numeratorName + " && " + denominatorName : !num ? numeratorName : denominatorName)
        << " is not TH1F. Quitting booking" << endl;

    return;

  }

  int nBinsNum = numH->GetNbinsX();
  int nBinsDen = denH->GetNbinsX();

  if (nBinsNum != nBinsDen) {
    edm::LogError("L1TEfficiencyPlotHandler") << " # bins in " << numeratorName << " and " << denominatorName
        << " are different. Quitting booking" << endl;
    return;
  }

  // double min = numH->GetXaxis()->GetXmin();
  // double max = numH->GetXaxis()->GetXmax();

  std::string denXaxisTitle{denH->GetXaxis()->GetTitle()};

  ibooker.setCurrentFolder(outputDir_);

  std::vector<float> bins;
  bins.reserve(nBinsNum+1);
  for(int i = 1 ; i <= nBinsNum ; ++i) {
    bins.push_back(numH->GetBinLowEdge(i));
  }
  bins.push_back(numH->GetBinLowEdge(nBinsNum)+numH->GetBinWidth(nBinsNum));
  float* binArray = &(bins[0]);

  h_efficiency_ = ibooker.book1D(plotName_, plotName_+";"+denXaxisTitle+";L1 efficiency", nBinsNum, binArray);

}

void L1TEfficiencyPlotHandler::computeEfficiency(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter)
{
  if (!h_efficiency_)
    return;

  edm::LogInfo("L1TEfficiencyPlotHandler") << " Computing efficiency for " << plotName_ << endl;

  MonitorElement *num = igetter.get(numeratorDir_ + "/" + plotName_ + numeratorSuffix_);
  MonitorElement *den = igetter.get(denominatorDir_ + "/" + plotName_ + denominatorSuffix_);

  TH1F *numH = num->getTH1F();
  TH1F *denH = den->getTH1F();

  numH->Sumw2();
  denH->Sumw2();

  TH1F *effH = h_efficiency_->getTH1F();

  effH->Divide(numH, denH, 1.0, 1.0, "B");
}

//___________DQM_analyzer_class________________________________________
L1TEfficiencyHarvesting::L1TEfficiencyHarvesting(const ParameterSet & ps) :
        verbose_(ps.getUntrackedParameter<bool>("verbose")),
        plotHandlers_()
{
  if (verbose_) {
    edm::LogInfo("L1TEfficiencyHarvesting") << "____________ Storage initialization ____________ " << endl;
  }

  for (auto plotConfig : ps.getUntrackedParameter < std::vector<edm::ParameterSet> > ("plotCfgs")) {
    vector < string > plots = plotConfig.getUntrackedParameter < vector < string >> ("plots");
    for (auto plot : plots) {
      plotHandlers_.push_back(L1TEfficiencyPlotHandler(plotConfig, plot));
    }
  }
}

//_____________________________________________________________________
L1TEfficiencyHarvesting::~L1TEfficiencyHarvesting()
{
}

//_____________________________________________________________________
void L1TEfficiencyHarvesting::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter)
{
  if (verbose_) {
    edm::LogInfo("L1TEfficiencyHarvesting") << "Called endRun." << endl;
  }

  for (auto plotHandler : plotHandlers_) {
    plotHandler.book(ibooker, igetter);
    plotHandler.computeEfficiency(ibooker, igetter);
  }
}

//_____________________________________________________________________
void L1TEfficiencyHarvesting::dqmEndLuminosityBlock(DQMStore::IGetter &igetter, LuminosityBlock const& lumiBlock,
    EventSetup const& c)
{
  if (verbose_) {
    edm::LogInfo("L1TEfficiencyHarvesting") << "Called endLuminosityBlock at LS=" << lumiBlock.id().luminosityBlock()
        << endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE (L1TEfficiencyHarvesting);
} //l1t
} // dqmoffline
