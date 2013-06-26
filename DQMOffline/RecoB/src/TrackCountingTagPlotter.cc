#include "DQMOffline/RecoB/interface/TrackCountingTagPlotter.h"
#include "DQMOffline/RecoB/interface/Tools.h"

using namespace std;
using namespace RecoBTag;

TrackCountingTagPlotter::TrackCountingTagPlotter(const std::string & tagName,
	const EtaPtBin & etaPtBin, const edm::ParameterSet& pSet, const bool& update, const unsigned int& mc, const bool& wf) :
  BaseTagInfoPlotter(tagName, etaPtBin), mcPlots_(mc), 
  nBinEffPur_(pSet.getParameter<int>("nBinEffPur")),
  startEffPur_(pSet.getParameter<double>("startEffPur")),
  endEffPur_(pSet.getParameter<double>("endEffPur")),
  willFinalize_(wf), lowerIPSBound(-35.0), upperIPSBound(35.0), finalized(false)
{
  const std::string dir(theExtensionString.substr(1));

  trkNbr3D = new FlavourHistograms<int>
	("selTrksNbr_3D" + theExtensionString, "Number of selected tracks for 3D IPS" + theExtensionString, 31, -0.5, 30.5,
	false, true, true, "b", update, dir, mc);

  trkNbr2D = new FlavourHistograms<int>
	("selTrksNbr_2D" + theExtensionString, "Number of selected tracks for 2D IPS" + theExtensionString, 31, -0.5, 30.5,
	false, true, true, "b", update, dir, mc);

  tkcntHistosSig3D[4] = new FlavourHistograms<double>
       ("ips_3D" + theExtensionString, "3D Significance of impact parameter",
	50, lowerIPSBound, upperIPSBound, false, true, true, "b", update, dir, mc) ;

  tkcntHistosSig3D[0] = new FlavourHistograms<double>
       ("ips1_3D" + theExtensionString, "3D Significance of impact parameter 1st trk",
	50, lowerIPSBound, upperIPSBound, false, true, true, "b", update, dir, mc) ;

  tkcntHistosSig3D[1] = new FlavourHistograms<double>
       ("ips2_3D" + theExtensionString, "3D Significance of impact parameter 2nd trk",
	50, lowerIPSBound, upperIPSBound, false, true, true, "b", update, dir, mc) ;

  tkcntHistosSig3D[2] = new FlavourHistograms<double>
       ("ips3_3D" + theExtensionString, "3D Significance of impact parameter 3rd trk",
	50, lowerIPSBound, upperIPSBound, false, true, true, "b", update, dir, mc) ;

  tkcntHistosSig3D[3] = new FlavourHistograms<double>
       ("ips4_3D" + theExtensionString, "3D Significance of impact parameter 4th trk",
	50, lowerIPSBound, upperIPSBound, false, true, true, "b", update, dir, mc) ;

  tkcntHistosSig2D[4] = new FlavourHistograms<double>
       ("ips_2D" + theExtensionString, "2D Significance of impact parameter",
	50, lowerIPSBound, upperIPSBound, false, true, true, "b", update, dir, mc) ;

  tkcntHistosSig2D[0] = new FlavourHistograms<double>
       ("ips1_2D" + theExtensionString, "2D Significance of impact parameter 1st trk",
	50, lowerIPSBound, upperIPSBound, false, true, true, "b", update, dir, mc) ;

  tkcntHistosSig2D[1] = new FlavourHistograms<double>
       ("ips2_2D" + theExtensionString, "2D Significance of impact parameter 2nd trk",
	50, lowerIPSBound, upperIPSBound, false, true, true, "b", update, dir, mc) ;

  tkcntHistosSig2D[2] = new FlavourHistograms<double>
       ("ips3_2D" + theExtensionString, "2D Significance of impact parameter 3rd trk",
	50, lowerIPSBound, upperIPSBound, false, true, true, "b", update, dir, mc) ;

  tkcntHistosSig2D[3] = new FlavourHistograms<double>
       ("ips4" + theExtensionString, "2D Significance of impact parameter 4th trk",
	50, lowerIPSBound, upperIPSBound, false, true, true, "b", update, dir, mc) ;

  if (willFinalize_) createPlotsForFinalize();

}


TrackCountingTagPlotter::~TrackCountingTagPlotter ()
{

  delete trkNbr3D;
  delete trkNbr2D;

  for(int n=0; n != 5; ++n) {
    delete tkcntHistosSig2D[n];
    delete tkcntHistosSig3D[n];
  }
  if (finalized) {
    for(int n=0; n != 4; ++n) delete effPurFromHistos[n];
  }
}

void TrackCountingTagPlotter::analyzeTag (const reco::BaseTagInfo * baseTagInfo,
	const int & jetFlavour)
{
  analyzeTag(baseTagInfo,jetFlavour,1.);
}

void TrackCountingTagPlotter::analyzeTag (const reco::BaseTagInfo * baseTagInfo,
					  const int & jetFlavour,
					  const float & w)
{

  const reco::TrackCountingTagInfo * tagInfo = 
	dynamic_cast<const reco::TrackCountingTagInfo *>(baseTagInfo);

  if (!tagInfo) {
    throw cms::Exception("Configuration")
      << "BTagPerformanceAnalyzer: Extended TagInfo not of type TrackCountingTagInfo. " << endl;
  }

  trkNbr3D->fill(jetFlavour, tagInfo->selectedTracks(0),w);
  trkNbr2D->fill(jetFlavour, tagInfo->selectedTracks(1),w);

  for(int n=0; n != tagInfo->selectedTracks(1) && n != 4; ++n)
    tkcntHistosSig2D[n]->fill(jetFlavour, tagInfo->significance(n,1),w);
  for(int n=tagInfo->selectedTracks(1); n < 4; ++n)
    tkcntHistosSig2D[n]->fill(jetFlavour, lowerIPSBound-1.0,w);

  for(int n=0; n != tagInfo->selectedTracks(0) && n != 4; ++n)
    tkcntHistosSig3D[n]->fill(jetFlavour, tagInfo->significance(n,0),w);
  for(int n=tagInfo->selectedTracks(0); n < 4; ++n)
    tkcntHistosSig3D[n]->fill(jetFlavour, lowerIPSBound-1.0,w);

  for(int n=0; n != tagInfo->selectedTracks(1); ++n)
    tkcntHistosSig2D[4]->fill(jetFlavour, tagInfo->significance(n,1),w);
  for(int n=0; n != tagInfo->selectedTracks(0); ++n)
    tkcntHistosSig3D[4]->fill(jetFlavour, tagInfo->significance(n,0),w);
}



void TrackCountingTagPlotter::createPlotsForFinalize (){
  //
  // final processing:
  // produce the misid. vs. eff histograms
  //
  const std::string dir("TrackCounting"+theExtensionString);

  effPurFromHistos[0] = new EffPurFromHistos (tkcntHistosSig3D[1],dir,mcPlots_,
		nBinEffPur_, startEffPur_,
		endEffPur_);
  effPurFromHistos[1] = new EffPurFromHistos (tkcntHistosSig3D[2],dir,mcPlots_,
		nBinEffPur_, startEffPur_,
		endEffPur_);
  effPurFromHistos[2] = new EffPurFromHistos (tkcntHistosSig2D[1],dir,mcPlots_,
		nBinEffPur_, startEffPur_,
		endEffPur_);
  effPurFromHistos[3] = new EffPurFromHistos (tkcntHistosSig2D[2],dir,mcPlots_,
		nBinEffPur_, startEffPur_,
		endEffPur_);
}

void TrackCountingTagPlotter::finalize ()
{
  for(int n=0; n != 4; ++n) effPurFromHistos[n]->compute();
  finalized = true;
}

void TrackCountingTagPlotter::psPlot(const std::string & name)
{
  const std::string cName("TrackCountingPlots"+ theExtensionString);
  setTDRStyle()->cd();
  TCanvas canvas(cName.c_str(), cName.c_str(), 600, 900);
  canvas.UseCurrentStyle();
  canvas.Divide(2,3);
  canvas.Print((name + cName + ".ps[").c_str());

  canvas.cd(1);
  trkNbr3D->plot();
  canvas.cd(2);
  tkcntHistosSig3D[4]->plot();
  for(int n=0; n < 4; n++) {
    canvas.cd(3+n);
    tkcntHistosSig3D[n]->plot();
  }

  canvas.Print((name + cName + ".ps").c_str());
  canvas.Clear();
  canvas.Divide(2,3);

  canvas.cd(1);
  trkNbr2D->plot();
  canvas.cd(2);
  tkcntHistosSig2D[4]->plot();
  for(int n=0; n != 4; ++n) {
    canvas.cd(3+n);
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


void TrackCountingTagPlotter::epsPlot(const std::string & name)
{
  trkNbr2D->epsPlot(name);
  trkNbr3D->epsPlot(name);
  for(int n=0; n != 5; ++n) {
    tkcntHistosSig2D[n]->epsPlot(name);
    tkcntHistosSig3D[n]->epsPlot(name);
  }
  if (finalized) {
    for(int n=0; n != 4; ++n) effPurFromHistos[n]->epsPlot(name);
  }
}
