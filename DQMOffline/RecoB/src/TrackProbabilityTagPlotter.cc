#include "DQMOffline/RecoB/interface/TrackProbabilityTagPlotter.h"
#include "DQMOffline/RecoB/interface/Tools.h"

using namespace std;
using namespace RecoBTag;

TrackProbabilityTagPlotter::TrackProbabilityTagPlotter(const std::string & tagName,
                               const EtaPtBin & etaPtBin, const edm::ParameterSet& pSet,
                               const unsigned int& mc, const bool& wf, DQMStore::IBooker & ibook) :
  BaseTagInfoPlotter(tagName, etaPtBin),
  nBinEffPur_(pSet.getParameter<int>("nBinEffPur")),
  startEffPur_(pSet.getParameter<double>("startEffPur")),
  endEffPur_(pSet.getParameter<double>("endEffPur")),
  mcPlots_(mc), willFinalize_(wf)
{
  const std::string dir(theExtensionString.substr(1));
  
  if (willFinalize_) return;
  
  for (unsigned int i = 1; i <= 4; i++) {
      tkcntHistosSig3D_.push_back(std::make_unique<FlavourHistograms<double>>
           ("ips" + std::to_string(i) + "_3D" + theExtensionString, "3D Probability of impact parameter " + std::to_string(i) + ". trk",
            50, -1.0, 1.0, false, true, true, "b", dir, mc, ibook));
  }

  tkcntHistosSig3D_.push_back(std::make_unique<FlavourHistograms<double>>
       ("ips_3D" + theExtensionString, "3D Probability of impact parameter",
        50, -1.0, 1.0, false, true, true, "b", dir, mc, ibook));

  
  for (unsigned int i = 1; i <= 4; i++) {
      tkcntHistosSig2D_.push_back(std::make_unique<FlavourHistograms<double>>
           ("ips" + std::to_string(i) + "_2D" + theExtensionString, "2D Probability of impact parameter " + std::to_string(i) + ". trk",
            50, -1.0, 1.0, false, true, true, "b", dir, mc, ibook));
  }

  tkcntHistosSig2D_.push_back(std::make_unique<FlavourHistograms<double>>
       ("ips_2D" + theExtensionString, "2D Probability of impact parameter",
        50, -1.0, 1.0, false, true, true, "b", dir, mc, ibook));
}


TrackProbabilityTagPlotter::~TrackProbabilityTagPlotter() { }

void TrackProbabilityTagPlotter::analyzeTag(const reco::BaseTagInfo * baseTagInfo,
                         double jec, 
                         int jetFlavour,
                         float w/*=1*/)
{
  const reco::TrackProbabilityTagInfo * tagInfo = 
    dynamic_cast<const reco::TrackProbabilityTagInfo *>(baseTagInfo);

  if (!tagInfo) {
    throw cms::Exception("Configuration")
      << "BTagPerformanceAnalyzer: Extended TagInfo not of type TrackProbabilityTagInfo. " << endl;
  }

  for (int n = 0; n != tagInfo->selectedTracks(1) && n != 4; ++n)
    tkcntHistosSig2D_[n]->fill(jetFlavour, tagInfo->probability(n, 1), w);
  for (int n = 0; n != tagInfo->selectedTracks(0) && n != 4; ++n)
    tkcntHistosSig3D_[n]->fill(jetFlavour, tagInfo->probability(n, 0), w);

  for (int n = 0; n != tagInfo->selectedTracks(1); ++n)
    tkcntHistosSig2D_[4]->fill(jetFlavour, tagInfo->probability(n, 1), w);
  for (int n = 0; n != tagInfo->selectedTracks(0); ++n)
    tkcntHistosSig3D_[4]->fill(jetFlavour, tagInfo->probability(n, 0), w);
}

void TrackProbabilityTagPlotter::finalize(DQMStore::IBooker & ibook, DQMStore::IGetter & igetter_)
{
  //
  // final processing:
  // produce the misid. vs. eff histograms
  //
  const std::string dir("TrackProbability" + theExtensionString);

  tkcntHistosSig3D_.clear();
  tkcntHistosSig2D_.clear();
  effPurFromHistos_.clear();
  
  for (unsigned int i = 2; i <= 3; i++) {
    tkcntHistosSig3D_.push_back(
            std::make_unique<FlavourHistograms<double>>
                    ("ips" + std::to_string(i) + "_3D" + theExtensionString, "3D Probability of impact parameter " + std::to_string(i) + ". trk",
                    50, -1.0, 1.0, "b", dir, mcPlots_, igetter_));
    effPurFromHistos_.push_back(
            std::make_unique<EffPurFromHistos>(*tkcntHistosSig3D_.back(), dir, mcPlots_, ibook, 
                nBinEffPur_, startEffPur_, endEffPur_));
  }

  for (unsigned int i = 2; i <= 3; i++) {
    tkcntHistosSig2D_.push_back(
            std::make_unique<FlavourHistograms<double>>
                    ("ips" + std::to_string(i) + "_2D" + theExtensionString, "2D Probability of impact parameter " + std::to_string(i) + ". trk",
                    50, -1.0, 1.0, "b", dir, mcPlots_, igetter_));
    effPurFromHistos_.push_back(
            std::make_unique<EffPurFromHistos>(*tkcntHistosSig2D_.back(), dir, mcPlots_, ibook, 
                nBinEffPur_, startEffPur_, endEffPur_));
  }

  for (int n = 0; n != 4; ++n) effPurFromHistos_[n]->compute(ibook);
}

void TrackProbabilityTagPlotter::psPlot(const std::string & name)
{
  const std::string cName("TrackProbabilityPlots"+ theExtensionString);
  setTDRStyle()->cd();
  TCanvas canvas(cName.c_str(), cName.c_str(), 600, 900);
  canvas.UseCurrentStyle();
  if (willFinalize_) {
    for (int n=0; n != 2; ++n) {
      canvas.Print((name + cName + ".ps").c_str());
      canvas.Clear();
      canvas.Divide(2,3);
      canvas.cd(1);
      effPurFromHistos_[0+n]->discriminatorNoCutEffic().plot();
      canvas.cd(2);
      effPurFromHistos_[0+n]->discriminatorCutEfficScan().plot();
      canvas.cd(3);
      effPurFromHistos_[0+n]->plot();
      canvas.cd(4);
      effPurFromHistos_[1+n]->discriminatorNoCutEffic().plot();
      canvas.cd(5);
      effPurFromHistos_[1+n]->discriminatorCutEfficScan().plot();
      canvas.cd(6);
      effPurFromHistos_[1+n]->plot();
    }
    return;
  }

  canvas.Clear();
  canvas.Divide(2,3);
  canvas.Print((name + cName + ".ps[").c_str());
  canvas.cd(1);

  tkcntHistosSig3D_[4]->plot();
  for (int n = 0; n != 4; ++n) {
    canvas.cd(2+n);
    tkcntHistosSig3D_[n]->plot();
  }

  canvas.Print((name + cName + ".ps").c_str());
  canvas.Clear();
  canvas.Divide(2,3);

  canvas.cd(1);
  tkcntHistosSig2D_[4]->plot();
  for (int n = 0; n != 4; ++n) {
    canvas.cd(2+n);
    tkcntHistosSig2D_[n]->plot();
  }

  canvas.Print((name + cName + ".ps").c_str());
  canvas.Print((name + cName + ".ps]").c_str());
}


void TrackProbabilityTagPlotter::epsPlot(const std::string & name)
{
  if (willFinalize_) {
    for (int n = 0; n != 4; ++n) effPurFromHistos_[n]->epsPlot(name);
    return;
  }
  for (int n = 0; n != 5; ++n) {
    tkcntHistosSig2D_[n]->epsPlot(name);
    tkcntHistosSig3D_[n]->epsPlot(name);
  }
}
