#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "TH1F.h"
#include "TH2F.h"

using namespace edm;
using namespace std;

class PlotCombiner : public DQMEDHarvester {
public:
  PlotCombiner(const edm::ParameterSet &pset);
  ~PlotCombiner() override;

private:
  void makePlot(const ParameterSet &pset, DQMStore::IBooker &, DQMStore::IGetter &);

  string myDQMrootFolder;
  const VParameterSet plots;

protected:
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob
};

PlotCombiner::PlotCombiner(const edm::ParameterSet &pset)
    : myDQMrootFolder(pset.getUntrackedParameter<string>("MyDQMrootFolder")),
      plots(pset.getUntrackedParameter<VParameterSet>("Plots")) {}

void PlotCombiner::dqmEndJob(DQMStore::IBooker &ibooker_, DQMStore::IGetter &igetter_) {
  for (VParameterSet::const_iterator pset = plots.begin(); pset != plots.end(); pset++) {
    makePlot(*pset, ibooker_, igetter_);
  }
}

void PlotCombiner::makePlot(const ParameterSet &pset, DQMStore::IBooker &ibooker_, DQMStore::IGetter &igetter_) {
  //get hold of MEs
  vector<string> inputMEnames = pset.getUntrackedParameter<vector<string> >("InputMEnames");
  vector<string> inputLabels = pset.getUntrackedParameter<vector<string> >("InputLabels");
  if (inputMEnames.size() != inputLabels.size()) {
    LogDebug("HLTriggerOfflineHeavyFlavor") << "Number of labels must match the histos[0]ber of InputMEnames" << endl;
    return;
  }
  vector<TH1 *> histos;
  vector<TString> labels;
  for (size_t i = 0; i < inputMEnames.size(); i++) {
    string MEname = myDQMrootFolder + "/" + inputMEnames[i];
    MonitorElement *ME = igetter_.get(MEname);
    if (ME == nullptr) {
      LogDebug("HLTriggerOfflineHeavyFlavor") << "Could not find ME: " << MEname << endl;
      continue;
    }
    histos.push_back(ME->getTH1());
    labels.push_back(inputLabels[i]);
  }
  if (histos.empty()) {
    return;
  }
  //figure out the output directory name
  string outputMEname = pset.getUntrackedParameter<string>("OutputMEname");
  ;
  string outputDir = myDQMrootFolder;
  string::size_type slashPos = outputMEname.rfind('/');
  if (string::npos != slashPos) {
    outputDir += "/" + outputMEname.substr(0, slashPos);
    outputMEname.erase(0, slashPos + 1);
  }
  ibooker_.setCurrentFolder(outputDir);
  //create output ME
  TH2F *output;
  if (histos[0]->GetXaxis()->GetXbins()->GetSize() == 0) {
    output = new TH2F(outputMEname.c_str(),
                      outputMEname.c_str(),
                      histos[0]->GetXaxis()->GetNbins(),
                      histos[0]->GetXaxis()->GetXmin(),
                      histos[0]->GetXaxis()->GetXmax(),
                      histos.size(),
                      0,
                      histos.size());
  } else {
    output = new TH2F(outputMEname.c_str(),
                      outputMEname.c_str(),
                      histos[0]->GetXaxis()->GetNbins(),
                      histos[0]->GetXaxis()->GetXbins()->GetArray(),
                      histos.size(),
                      0,
                      histos.size());
  }
  output->SetTitle(outputMEname.c_str());
  output->SetXTitle(histos[0]->GetXaxis()->GetTitle());
  output->SetStats(kFALSE);
  output->SetOption("colztexte");
  for (size_t i = 0; i < histos.size(); i++) {
    for (int j = 1; j <= histos[0]->GetNbinsX(); j++) {
      output->SetBinContent(j, i + 1, histos[i]->GetBinContent(j));
      output->SetBinError(j, i + 1, histos[i]->GetBinError(j));
    }
    output->GetYaxis()->SetBinLabel(i + 1, labels[i]);
  }
  ibooker_.book2D(outputMEname, output);
  delete output;
}

PlotCombiner::~PlotCombiner() {}

//define this as a plug-in
DEFINE_FWK_MODULE(PlotCombiner);
