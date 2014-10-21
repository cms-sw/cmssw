#include "DQMOffline/RecoB/interface/TrackProbabilityTagPlotter.h"
#include "DQMOffline/RecoB/interface/Tools.h"

using namespace std;
using namespace RecoBTag;

TrackProbabilityTagPlotter::TrackProbabilityTagPlotter(const std::string & tagName,
						       const EtaPtBin & etaPtBin, const edm::ParameterSet& pSet,
						       const bool& update, const unsigned int& mc, const bool& wf, DQMStore::IBooker & ibook) :
  BaseTagInfoPlotter(tagName, etaPtBin),
  nBinEffPur_(pSet.getParameter<int>("nBinEffPur")),
  startEffPur_(pSet.getParameter<double>("startEffPur")),
  endEffPur_(pSet.getParameter<double>("endEffPur")),
  finalized(false), mcPlots_(mc), willFinalize_(wf), ibook_(ibook)
{
  const std::string dir(theExtensionString.substr(1));

  tkcntHistosSig3D[4] = new FlavourHistograms<double>
       ("ips_3D" + theExtensionString, "3D Probability of impact parameter",
	50, -1.0, 1.0, false, true, true, "b", update,dir, mc, ibook) ;

  tkcntHistosSig3D[0] = new FlavourHistograms<double>
       ("ips1_3D" + theExtensionString, "3D Probability of impact parameter 1st trk",
	50, -1.0, 1.0, false, true, true, "b", update,dir, mc, ibook) ;

  tkcntHistosSig3D[1] = new FlavourHistograms<double>
       ("ips2_3D" + theExtensionString, "3D Probability of impact parameter 2nd trk",
	50, -1.0, 1.0, false, true, true, "b", update,dir, mc, ibook) ;

  tkcntHistosSig3D[2] = new FlavourHistograms<double>
       ("ips3_3D" + theExtensionString, "3D Probability of impact parameter 3rd trk",
	50, -1.0, 1.0, false, true, true, "b", update,dir, mc, ibook) ;

  tkcntHistosSig3D[3] = new FlavourHistograms<double>
       ("ips4_3D" + theExtensionString, "3D Probability of impact parameter 4th trk",
	50, -1.0, 1.0, false, true, true, "b", update,dir, mc, ibook) ;

  tkcntHistosSig2D[4] = new FlavourHistograms<double>
       ("ips_2D" + theExtensionString, "2D Probability of impact parameter",
	50, -1.0, 1.0, false, true, true, "b", update,dir, mc, ibook) ;

  tkcntHistosSig2D[0] = new FlavourHistograms<double>
       ("ips1_2D" + theExtensionString, "2D Probability of impact parameter 1st trk",
	50, -1.0, 1.0, false, true, true, "b", update,dir, mc, ibook) ;

  tkcntHistosSig2D[1] = new FlavourHistograms<double>
       ("ips2_2D" + theExtensionString, "2D Probability of impact parameter 2nd trk",
	50, -1.0, 1.0, false, true, true, "b", update,dir, mc, ibook) ;

  tkcntHistosSig2D[2] = new FlavourHistograms<double>
       ("ips3_2D" + theExtensionString, "2D Probability of impact parameter 3rd trk",
	50, -1.0, 1.0, false, true, true, "b", update,dir, mc, ibook) ;

  tkcntHistosSig2D[3] = new FlavourHistograms<double>
       ("ips4" + theExtensionString, "2D Probability of impact parameter 4th trk",
	50, -1.0, 1.0, false, true, true, "b", update,dir, mc, ibook) ;

  if (willFinalize_) createPlotsForFinalize(ibook);
 
}


TrackProbabilityTagPlotter::~TrackProbabilityTagPlotter ()
{

  for(int n=0; n != 5; ++n) {
    delete tkcntHistosSig2D[n];
    delete tkcntHistosSig3D[n];
  }
  if (finalized) {
    for(int n=0; n != 4; ++n) delete effPurFromHistos[n];
  }
}


void TrackProbabilityTagPlotter::analyzeTag (const reco::BaseTagInfo * baseTagInfo,
					     const double & jec, 
					     const int & jetFlavour,
					     const float & w)
{
  const reco::TrackProbabilityTagInfo * tagInfo = 
	dynamic_cast<const reco::TrackProbabilityTagInfo *>(baseTagInfo);

  if (!tagInfo) {
    throw cms::Exception("Configuration")
      << "BTagPerformanceAnalyzer: Extended TagInfo not of type TrackProbabilityTagInfo. " << endl;
  }

  for(int n=0; n != tagInfo->selectedTracks(1) && n != 4; ++n)
    tkcntHistosSig2D[n]->fill(jetFlavour, tagInfo->probability(n,1),w);
  for(int n=0; n != tagInfo->selectedTracks(0) && n != 4; ++n)
    tkcntHistosSig3D[n]->fill(jetFlavour, tagInfo->probability(n,0),w);

  for(int n=0; n != tagInfo->selectedTracks(1); ++n)
    tkcntHistosSig2D[4]->fill(jetFlavour, tagInfo->probability(n,1),w);
  for(int n=0; n != tagInfo->selectedTracks(0); ++n)
    tkcntHistosSig3D[4]->fill(jetFlavour, tagInfo->probability(n,0),w);
}

void TrackProbabilityTagPlotter::analyzeTag (const reco::BaseTagInfo * baseTagInfo, const double & jec, 
					     const int & jetFlavour)
{
  analyzeTag(baseTagInfo,jetFlavour,1.);
}


void TrackProbabilityTagPlotter::createPlotsForFinalize(DQMStore::IBooker & ibook){
  const std::string dir("TrackProbability"+theExtensionString);

  effPurFromHistos[0] = new EffPurFromHistos (tkcntHistosSig3D[1],dir,mcPlots_, ibook, 
					      nBinEffPur_, startEffPur_, endEffPur_);
  effPurFromHistos[1] = new EffPurFromHistos (tkcntHistosSig3D[2],dir,mcPlots_, ibook, 
					      nBinEffPur_, startEffPur_, endEffPur_);
  effPurFromHistos[2] = new EffPurFromHistos (tkcntHistosSig2D[1],dir,mcPlots_, ibook, 
					      nBinEffPur_, startEffPur_, endEffPur_);
  effPurFromHistos[3] = new EffPurFromHistos (tkcntHistosSig2D[2],dir,mcPlots_, ibook, 
					      nBinEffPur_, startEffPur_, endEffPur_);  
}

void TrackProbabilityTagPlotter::finalize ()
{
  //
  // final processing:
  // produce the misid. vs. eff histograms
  //
  for(int n=0; n != 4; ++n) effPurFromHistos[n]->compute(ibook_);
  finalized = true;
}

void TrackProbabilityTagPlotter::psPlot(const std::string & name)
{
  const std::string cName("TrackProbabilityPlots"+ theExtensionString);
  setTDRStyle()->cd();
  TCanvas canvas(cName.c_str(), cName.c_str(), 600, 900);
  canvas.UseCurrentStyle();
  canvas.Divide(2,3);
  canvas.Print((name + cName + ".ps[").c_str());
  canvas.cd(1);

  tkcntHistosSig3D[4]->plot();
  for(int n=0; n != 4; ++n) {
    canvas.cd(2+n);
    tkcntHistosSig3D[n]->plot();
  }

  canvas.Print((name + cName + ".ps").c_str());
  canvas.Clear();
  canvas.Divide(2,3);

  canvas.cd(1);
  tkcntHistosSig2D[4]->plot();
  for(int n=0; n != 4; ++n) {
    canvas.cd(2+n);
    tkcntHistosSig2D[n]->plot();
  }

  if (finalized) {
    for(int n=0; n != 2; ++n) {
      canvas.Print((name + cName + ".ps").c_str());
      canvas.Clear();
      canvas.Divide(2,3);
      canvas.cd(1);
      effPurFromHistos[0+n]->discriminatorNoCutEffic()->plot();
      canvas.cd(2);
      effPurFromHistos[0+n]->discriminatorCutEfficScan()->plot();
      canvas.cd(3);
      effPurFromHistos[0+n]->plot();
      canvas.cd(4);
      effPurFromHistos[1+n]->discriminatorNoCutEffic()->plot();
      canvas.cd(5);
      effPurFromHistos[1+n]->discriminatorCutEfficScan()->plot();
      canvas.cd(6);
      effPurFromHistos[1+n]->plot();
    }
  }

  canvas.Print((name + cName + ".ps").c_str());
  canvas.Print((name + cName + ".ps]").c_str());
}


void TrackProbabilityTagPlotter::epsPlot(const std::string & name)
{
  for(int n=0; n != 5; ++n) {
    tkcntHistosSig2D[n]->epsPlot(name);
    tkcntHistosSig3D[n]->epsPlot(name);
  }
  if (finalized) {
    for(int n=0; n != 4; ++n) effPurFromHistos[n]->epsPlot(name);
  }
}
